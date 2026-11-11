# PCIe / PLX / IOMMU audit and P2P matrix, par1-cs25 after the 2026-11-05 re-cabling

Date: 2026-11-06. Read-only audit: no host configuration, kernel command line, sysctl or BIOS setting was changed.
Box: par1-cs25 (HUANANZHI H12D-8D v2.0, AMI BIOS 2.2 of 2026-01-06, EPYC 7K62 single socket, one NUMA node),
kernel `7.1.7-100.p2p_post7.fc43`, booted 2026-11-05 14:03 (uptime 18 h at capture).
Probe sources: `tools/rdna/p2p_probe/` (this branch; build line in the header of `p2p_probe.cpp`). Raw logs on the box: `~/w4a8_runs/pcie_audit/` (`lspci_vvv.txt`,
`jk_b*.txt` journal per boot, `p2p/w7_win*.log`, `p2p/w8_win*.log`, `p2p/rccl8/*.log`).

## Summary

| Item | Result |
|---|---|
| Links | Every V620, PLX port and root port trains at 16 GT/s x16; MPS 256 end to end; ASPM off. No downtrain. |
| Errors, current boot | None: no `[Hardware Error]`/GHES event, no IOMMU fault, no sync flood in 18 h. |
| AER | Firmware-first: `_OSC` says the platform does not grant AER/DPC/LTR to the OS, so the sysfs `aer_*` counters are always 0 and say nothing. Only GHES lines and reset reasons show errors. |
| Error history | Boots on the old cabling logged BadTLP/Timeout/CmplAbort corrected errors (one port 15,769 times) and a fatal completion timeout at root port 80:01.1 ending in a data-fabric sync flood reset (2026-11-05 02:44, again 02:49). Those ports no longer exist. |
| ACS | PLX downstream ports: all ACS control bits clear (`disable_acs_redir` works, intra-switch P2P is direct). Root ports 00:01.1 and c0:03.1 keep SrcValid+ReqRedir+CmplRedir+UpstreamFwd (0x1d). |
| IOMMU | Default domain `Passthrough`; every group is `identity`. No translation on the P2P path. Groups follow the PLX tree (V620 pairs share groups), irrelevant under identity. |
| P2P enabled | KFD reports P2P for all 9 peers of every GPU (flags 3 = enabled, non-coherent; no NO_PEER_TO_PEER bit); `amdgpu.pcie_p2p=Y`. |
| P2P write (push) | 24.5-27 GB/s per direction for every pair; cross-root 25.2 = intra-switch 24.6. Latency one-way flag 6.2 us intra, 6.9 us cross. |
| P2P read (pull) | Intra 24.4 GB/s, **cross-root 5.7 GB/s**. Never pull across roots. |
| Cross-root aggregate | Per direction: 1 pair 25.2, 2 pairs 27.6, **3 pairs 18.7, 4 pairs 17.0 GB/s**. More concurrent distinct cross-root pairs make it slower. Latency regime (<= 64 KiB) shows no collapse. |
| Interference | Intra-switch traffic on other GPUs does not slow cross-root traffic and vice versa. Switches are independent. |
| RCCL TP8 | One ring in HIP order (2 cross-root hops), all `P2P/direct pointer`; `NCCL_P2P_LEVEL` pxb/sys/phb give identical rings and numbers. 115 us at 4 KiB, 24.5 GB/s busBW at 8 MiB. |

**Update 2026-11-07:** the tables above describe the original command line (`amd_iommu=on iommu=pt`, firmware-first AER). The four
proposals were tested; the box now runs `amd_iommu=on` removed and `pcie_ports=native` added (kernel-owned AER/DPC, live `aer_*`
counters, P2P and RCCL numbers unchanged). Root-port ACS clear and `iommu=off` change nothing. See "Results of the tests".

## Topology (since 2026-11-05)

Two PEX88096 (PEX880xx) switches, each behind one x16 Gen4 root port, one W7800 + four V620 each.
Inside a switch the five endpoints hang off nested virtual switches (see the tree); each GPU also has its own
internal Navi10 upstream/downstream bridge pair.

| Root | Root port | PLX upstream | Endpoints (HIP order, bus) | KFD/rocm-smi GPU |
|---|---|---|---|---|
| 00 (slot B) | 00:01.1 | 01:00.0 | W7800 07 (HIP 5), V620 0a, 0f, 14, 17 (HIP 6-9) | GPU0 07, GPU1-4 0a/0f/14/17 |
| c0 (slot A) | c0:03.1 | c5:00.0 | W7800 cb (HIP 0), V620 ce, d3, d8, db (HIP 1-4) | GPU5 cb, GPU6-9 ce/d3/d8/db |

Nesting (slot B; slot A is identical with c5/c6/c7.. buses): `00:01.1 > 01:00.0 > {02:00.0 > 03:00.0 > {04:00.0 W7800, 04:10.0 0a},
02:04.0 > 0b:00.0 > 0c:00.0 > 0f, 02:08.0 > 10:00.0 > {11:00.0 14, 11:10.0 17}, 02:0c/02:1c mgmt}`. V620 pairs
(14,17) and (d8,db) share a second-level virtual switch; measured latency and bandwidth show no difference for them.
rocm-smi `--showtopo{access,hops,weight,type}` report every pair as PCIE, weight 40, 2 hops, access True: it does not
distinguish intra-switch from cross-root, so it cannot be used to place ranks. The 2026-09-23 topology file
(`docs/topology/build_server_topology_2026-09-23.txt`) had the same two-switch layout on roots 00 and 40, all rocm-smi
pairs also weight 40 / 2 hops; the second switch moved from root 40 to root c0 (c0 also carries 2 NICs and 2 NVMe).

## Links, MPS, ACS (lspci -vvv, 2026-11-06 08:29)

| Device | BDF | LnkCap | LnkSta | MPS/MRRS | ACS control (set bits) | RO / ExtTag / 10-bit tag req |
|---|---|---|---|---|---|---|
| Root port (slot B) | 00:01.1 | 16GT x16 | 16GT x16 | 256/512 | SrcValid, ReqRedir, CmplRedir, UpstreamFwd | on / on / off |
| Root port (slot A) | c0:03.1 | 16GT x16 | 16GT x16 | 256/512 | SrcValid, ReqRedir, CmplRedir, UpstreamFwd | on / on / off |
| PLX upstream | 01:00.0, c5:00.0 | 16GT x16 | 16GT x16 | 256/128 | none (no ACS cap on upstream) | on / off / off |
| PLX downstream | 02:00/04/08/0c.0, 04:10.0, 0c:00.0, 11:00.0, 11:10.0 (and c6:*, c8:10.0, d0:00.0, d5:*) | 16GT x16 | 16GT x16 | 256/128 | **none** (ACSCap 0x207f, ctl 0x0000) | on / off / off |
| GPU internal downstream bridge | 09:00.0, 0e:00.0, 13:00.0, 16:00.0 (and cd/d2/d7/da) | 16GT x16 | 16GT x16 | 256/512 | SrcValid, ReqRedir, CmplRedir, UpstreamFwd | on / on / off |
| V620 | 0a, 0f, 14, 17, ce, d3, d8, db | 16GT x16 | 16GT x16 | 256/512 | none | on / on / off |
| W7800 | 07, cb | 16GT x16 | 16GT x16 (see note) | 256/512 | none | on / on / off |

- All eight V620 are on 16 GT/s x16 with ASPM disabled on every port; DevSta shows sticky CorrErr+/UnsupReq+ on switch and GPU
  ports (status bits, not counters; UnsupReq is masked in the uncorrectable-error mask).
- Note on W7800: the capture caught both W7800 in runtime-suspend (D3), where the switch port reports 2.5 GT/s x1
  (`04:00.0`, `c8:00.0`). `rocm-smi` wakes them and the link then reads 16 GT/s x16. Not a cabling defect. The retained dmesg (from t=2932 s, the ring
  buffer had already wrapped) shows `Configured ACS to 0x0000` for the W7800 ports 44 times (04:00.0) and 45 times (c8:00.0)
  plus 15 times each for the unused ports 02:0c.0 and c6:0c.0 [INFERENCE: one per W7800 resume or hot-reset of the port].
- The root complex does not complete 10-bit tags (`10BitTagComp-` on 00:01.1 and c0:03.1), so 10-bit tags cannot be enabled
  for DMA to host. Relaxed ordering is already on at every port. Max payload is limited to 256 B by the GPUs.

## Error and fault history (`journalctl -k`, GHES/APEI lines, since 2026-10-27)

| Boot | Span | Events | Content |
|---|---|---|---|
| -5 | 10-27 23:28 to 11-01 13:15 | 53 corrected | BadTLP at 40:03.1 (15), Replay timeout at 41:00.0, AdvNonFatal+UnsupReq on switch/GPU ports (old cabling) |
| -4 | 11-01 13:17 to 20:27 | 15,825 corrected | 15,769 on 83:00.0 (cor 0x2000, uncor CmpltAbrt 0x8000); old cabling |
| -3 | 11-01 20:29 to 11-05 02:44 | 315 corrected | 291 BadTLP at 40:03.1; ended in a sync-flood reset |
| -2 | 11-05 02:47 to 02:49 | reset reason | `Previous system reset reason [0x08000800]: an uncorrected error caused a data fabric sync flood`; fatal record: root port 80:01.1 secondary bus 0x81, uncor status 0x4000 (completion timeout); also one corrected DIMM P0D0 storage error |
| -1 | 11-05 02:51 to 13:47 | 3 corrected + same reset record | 3 BadTLP at 40:03.1 (08:38, 09:42, 09:47); the 80:01.1 fatal record is re-logged at boot |
| 0 | 11-05 14:03 to now | **0** | no GHES event, no AER line, no IOMMU fault, no `Hardware Error` |

Root ports 80:01.1 and 40:03.1 do not exist in the current tree (old cabling). The sysfs `aer_dev_*` and `aer_rootport_*`
counters are all 0 on every device, which is uninformative: `_OSC` leaves AER with the firmware (`OS now controls [PCIeHotplug
PME PCIeCapability]`). The 2026-09-24 PCI SERR host crash fits the same pattern (fatal PCIe error, firmware sync flood) [INFERENCE].
Kernel cmdline note: `amd_iommu=on` logs `AMD-Vi: Unknown option - 'on'` (ignored; IOMMU is enabled by default), the effective
modes are `iommu=pt` (`iommu: Default domain type: Passthrough`), interrupt remapping and x2APIC on.

## IOMMU groups (all `identity`, 83 groups)

| Group | Members |
|---|---|
| 73 | W7800 07, V620 0a and the PLX path (02:00.0 to 09:00.0) |
| 74 | V620 0f |
| 75 | V620 14, V620 17 |
| 19 | W7800 cb, V620 ce |
| 20 | V620 d3 |
| 21 | V620 d8, V620 db |

`pcie_acs_override=downstream,multifunction` only splits groups for VFIO purposes; with `iommu=pt` no device is translated, so
the grouping has no effect on P2P or DMA. The runtime ACS state is set by the kernel at probe (`Configured ACS to 0x0000` on the
PLX ports, `disable_acs_redir=pci:1000:c010`).

## amdgpu / KFD

`amdgpu.pcie_p2p=Y`, `pcie_gen_cap=0`, `pcie_lane_cap=0`, `ppfeaturemask=0xfff7ffff`, `runpm=-1`. KFD p2p_links:
each GPU lists 9 peers, `weight 40`, `max_bandwidth 32000`, `flags 3`; `hipDeviceGetP2PAttribute`: access=1,
perf_rank=2, native_atomics=1 for all pairs. `pp_dpm_pcie` of a V620: state 0 2.5GT x1, state 1 16GT x16 (current).
`rocm-bandwidth-test` is not installed; `rccl-tests` exist at `~/rccl-tests-build/build/` and were used.

## P2P matrix (8 V620, HIP 1-4 + 6-9)

Method: `p2p_probe` (kernel stores/loads on peer VRAM, hipMemcpyPeerAsync, flag ping-pong). `push` = a kernel on the source
GPU stores into peer VRAM (what rdna_ar does; the stage is `hipDeviceMallocUncached` like the production staging buffer);
`pull` = a kernel on the destination reads peer VRAM; 128 blocks x 256 threads; 0.4 s per cell; every cell verifies the
landed data. One pair at a time unless stated. Windows: 7-GPU runs 08:35-08:52, 8-GPU runs 08:54-09:05, no server up,
no KFD client other than the probe. GPU temperatures 32-43 C at start.

### Single pair, GB/s per direction (min / mean / max over ordered pairs)

| Mode | Size | Intra A (ce,d3,d8,db) | Intra B (0a,0f,14,17) | Cross-root |
|---|---|---|---|---|
| push (kernel stores) | 64 KiB | 24.6 / 25.2 / 27.0 | 24.6 / 24.6 / 24.6 | 25.2 / 25.4 / 26.8 |
| push (kernel stores) | 1024 KiB | 24.6 / 25.2 / 26.9 | 24.5 / 24.6 / 24.6 | 25.1 / 25.4 / 26.8 |
| push (kernel stores) | 5120 KiB | 24.0 / 24.6 / 26.3 | 24.0 / 24.0 / 24.0 | 24.9 / 25.1 / 26.2 |
| push (kernel stores) | 65536 KiB | 25.4 / 26.1 / 28.0 | 25.4 / 25.4 / 25.4 | 25.5 / 25.8 / 27.8 |
| pull (kernel loads) | 64 KiB | 24.4 / 25.1 / 27.0 | 24.4 / 24.4 / 24.4 | 5.7 / 5.7 / 5.8 |
| pull (kernel loads) | 1024 KiB | 24.4 / 25.0 / 27.0 | 24.4 / 24.4 / 24.4 | 5.6 / 5.7 / 5.8 |
| pull (kernel loads) | 5120 KiB | 23.7 / 24.4 / 26.2 | 23.8 / 23.8 / 23.8 | 5.6 / 5.6 / 5.8 |
| pull (kernel loads) | 65536 KiB | 25.4 / 26.1 / 28.2 | 25.4 / 25.4 / 25.4 | 5.7 / 5.8 / 5.9 |
| hipMemcpyPeerAsync | 64 KiB | 6.0 / 6.1 / 6.1 | 5.9 / 5.9 / 6.0 | 4.6 / 4.7 / 4.8 |
| hipMemcpyPeerAsync | 1024 KiB | 19.9 / 20.4 / 21.5 | 19.9 / 19.9 / 20.0 | 19.8 / 20.2 / 21.6 |
| hipMemcpyPeerAsync | 5120 KiB | 24.2 / 24.9 / 26.7 | 24.2 / 24.3 / 24.3 | 24.2 / 24.4 / 25.3 |
| hipMemcpyPeerAsync | 65536 KiB | 25.5 / 26.1 / 28.0 | 25.4 / 25.5 / 25.5 | 25.5 / 25.6 / 26.4 |

Push into `db` is about 8 % faster (26.9 vs 24.6 GB/s at 1 MiB) from any source, including cross-root; cause not
investigated. Bidirectional pair (both directions at once, total): intra 52.5, cross-root 53.5 GB/s at 1 MiB; 54.8 / 55.1 at 64 MiB.
Cross-root pulls are 4.4x slower than pushes; hipMemcpyPeerAsync cross-root ties intra above 5 MiB.

### Push 1 MiB, GB/s (src rows, dst columns)

| src \ dst | ce | d3 | d8 | db | 0a | 0f | 14 | 17 |
|---|---|---|---|---|---|---|---|---|
| ce | - | 24.6 | 24.6 | 26.9 | 25.2 | 25.2 | 25.2 | 25.2 |
| d3 | 24.6 | - | 24.6 | 26.9 | 25.2 | 25.2 | 25.2 | 25.2 |
| d8 | 24.6 | 24.6 | - | 26.9 | 25.2 | 25.2 | 25.2 | 25.2 |
| db | 24.6 | 24.6 | 24.6 | - | 25.2 | 25.2 | 25.2 | 25.2 |
| 0a | 25.2 | 25.1 | 25.1 | 26.8 | - | 24.5 | 24.6 | 24.6 |
| 0f | 25.1 | 25.2 | 25.1 | 26.8 | 24.5 | - | 24.6 | 24.6 |
| 14 | 25.2 | 25.2 | 25.2 | 26.8 | 24.6 | 24.6 | - | 24.6 |
| 17 | 25.2 | 25.2 | 25.2 | 26.8 | 24.6 | 24.6 | 24.6 | - |

### Latency, one-way flag ping-pong (us; 1 block, payload pushed before the flag, 2000 iterations, best of 3)

Flag only (payload 0):

| a \ b | ce | d3 | d8 | db | 0a | 0f | 14 | 17 |
|---|---|---|---|---|---|---|---|---|
| ce | - | 6.23 | 6.27 | 6.14 | 6.93 | 6.94 | 6.89 | 6.91 |
| d3 | 6.23 | - | 6.28 | 6.25 | 6.93 | 6.94 | 6.90 | 6.89 |
| d8 | 6.27 | 6.28 | - | 6.20 | 6.95 | 6.92 | 6.90 | 6.88 |
| db | 6.14 | 6.25 | 6.20 | - | 6.86 | 6.86 | 6.81 | 6.79 |
| 0a | 6.93 | 6.93 | 6.95 | 6.86 | - | 6.33 | 6.27 | 6.25 |
| 0f | 6.94 | 6.94 | 6.92 | 6.86 | 6.33 | - | 6.23 | 6.26 |
| 14 | 6.89 | 6.90 | 6.90 | 6.81 | 6.27 | 6.23 | - | 6.23 |
| 17 | 6.91 | 6.89 | 6.88 | 6.79 | 6.25 | 6.26 | 6.23 | - |

Payload 64 KiB before the flag:

| a \ b | ce | d3 | d8 | db | 0a | 0f | 14 | 17 |
|---|---|---|---|---|---|---|---|---|
| ce | - | 8.35 | 8.36 | 8.15 | 9.03 | 9.06 | 8.85 | 8.83 |
| d3 | 8.35 | - | 8.37 | 8.19 | 9.02 | 9.04 | 8.86 | 8.85 |
| d8 | 8.36 | 8.37 | - | 8.11 | 9.02 | 9.01 | 8.84 | 8.83 |
| db | 8.15 | 8.19 | 8.11 | - | 9.04 | 9.03 | 8.70 | 8.65 |
| 0a | 9.03 | 9.02 | 9.02 | 9.04 | - | 8.68 | 8.59 | 8.56 |
| 0f | 9.06 | 9.04 | 9.01 | 9.03 | 8.68 | - | 8.58 | 8.58 |
| 14 | 8.85 | 8.86 | 8.84 | 8.70 | 8.59 | 8.58 | - | 8.37 |
| 17 | 8.83 | 8.85 | 8.83 | 8.65 | 8.56 | 8.58 | 8.37 | - |

Intra-switch 6.1-6.3 us flag-only (7.0 at 16 KiB, 8.4 at 64 KiB), cross-root 6.8-6.9 (7.6, 9.0): the root hop costs +0.7 us.

## Concurrent flows (push, GB/s; size independent between 64 KiB and 20 MiB)

Cross-root, per direction (A to B unless noted; the B to A and mirrored placements match; placement of the pairs does not matter):

| Concurrent cross-root pairs | Aggregate | Per pair | Bidirectional (both directions), aggregate |
|---|---|---|---|
| 1 | 25.2 | 25.2 | 53.5 |
| 2 | 27.6 | 13.8 | 55.1 |
| 3 | 18.7 | 6.2 | 39.5 (6.6 each way) |
| 4 | 17.0 | 4.2 | 35.3 (4.4 each way) |
| 4 sources to 1 destination (fan-in) | 25.6 | 6.4 | |
| 1 source to 3-4 destinations (fan-out) | 27.7 | 6.9-9.2 | |

Not changed by the block count (8 to 512) or by which GPUs are chosen (every 2-subset gives 27.6, every 3-subset 18.3-19.6).

Intra-switch, 4 GPUs concurrently:

| Pattern | Aggregate | Per flow |
|---|---|---|
| ring ce>d3>d8>db>ce (4 flows) | 105 | 26.3 |
| all-to-all inside one switch (12 flows) | 105-107 (64 KiB), 93 (1-20 MiB) | 8.9 / 7.7 |
| both switches, ring each (8 flows) | 210 | 26.3 |
| both switches, all-to-all (24 flows) | 207 (64 KiB) | 8.8 |

A GPU sources at most about 26-27 GB/s (egress limit); the intra-switch fabric is non-blocking.

Interference (disjoint endpoints): cross-root pair(s) on GPU 0 (and 1) with intra-switch rings/swaps running on the
remaining GPUs of both switches:

| Foreground | Alone | With intra background |
|---|---|---|
| 1 cross pair (64 KiB / 1 MiB / 5 MiB / 20 MiB) | 25.2 / 25.2 / 25.0 / 25.4 | 25.1 / 25.1 / 24.8 / 25.4 |
| 2 cross pairs | 27.6 / 27.6 / 27.5 / 27.8 | 27.1 / 27.1 / 26.8 / 27.6 |
| intra rings (the background itself, 5-6 flows) | 127.5-135.3 total | 125.4-134.7 total, unchanged |

## Latency regime for a hierarchical collective (us, 8 GPUs, median of 3 runs)

k cross pairs A_i to B_i fired together; "push+flag" is the one-way completion; "exchange" is both directions at once (one-shot
step: push, fence, flag, wait for the peer's flag).

| Payload | push+flag k=1 | k=2 | k=3 | k=4 | exchange k=1 | k=2 | k=3 | k=4 |
|---|---|---|---|---|---|---|---|---|
| 4 KiB | 7.2 | 7.2 | 7.1 | 7.0 | 7.4 | 7.2 | 7.2 | 7.2 |
| 8 KiB | 6.9 | 7.3 | 7.2 | 7.2 | 6.8 | 7.7 | 7.5 | 7.4 |
| 16 KiB | 7.1 | 7.4 | 7.4 | 7.6 | 7.1 | 7.6 | 8.1 | 8.0 |
| 32 KiB | 7.7 | 8.0 | 7.8 | 7.7 | 7.7 | 8.0 | 7.8 | 7.8 |
| 64 KiB | 9.0 | 9.2 | 10.3 | 10.1 | 9.9 | 10.1 | 10.6 | 13.8 |

One-shot all-to-all step (every GPU pushes its payload to all peers and waits for all flags, no reduction):

| Payload | 8 GPUs flat (7 targets) | 4 GPUs inside each switch (both at once) |
|---|---|---|
| 4 KiB | 15.7 | 10.4 |
| 8 KiB | 13.6 | 11.3 |
| 16 KiB | 20.4 | 10.9 |
| 32 KiB | 36.8 | 14.8 |
| 64 KiB | 62.4 | 22.0 |

## RCCL (rccl-tests `all_reduce_perf`, system ROCm 7.14 RCCL, one process, `-g N`, 20 iterations)

Env: `NCCL_PROTO=Simple RCCL_P2P_NET_DISABLE=1 RCCL_P2P_BATCH_ENABLE=1 RCCL_MSCCL_ENABLE=0`, plus `NCCL_P2P_LEVEL`.
Graph (TP8): one ring 0..7 in HIP order (ce,d3,d8,db,0a,0f,14,17), chain trees, 2 channels, only db to 0a and 17 to ce cross a root,
all links `via P2P/direct pointer`, `isAllDirectP2p 1`. NCCL topology reports intra-switch PXB (6/24.0) and cross-root PHB (8/24.0), but
`pxb`, `sys` and `phb` produce identical graphs and timings (RCCL does not drop the cross-root P2P at `pxb`).

| Size | TP8 us (busBW GB/s) | TP4 intra A us | TP4 2+2 cross us | TP2 intra us | TP2 cross-root us |
|---|---|---|---|---|---|
| 4 KiB | 115 (0.06) | 52.6 | 55.6 | 42.8 (first call) / 28 | 52 / 29 |
| 8 KiB | 114 (0.13) | 51.9 | 54.8 | 23.6 | 25.1 |
| 64 KiB | 182 (0.6) | 58.0 | 60.1 | 28.2 | 29.0 |
| 1 MiB | 235 (7.8) / 180 (10.2) | 101 | 96 | 64.6 | 64.2 |
| 4 MiB | 331 (22.2) | 283 | 278 | 188 | 191 |
| 8 MiB | 598 (24.5) | 506 | 517 | 362 | 357 |

The 1 MiB TP8 row shows the two measured passes (out-of-place / in-place). TP2 across roots costs the same as TP2 inside a switch
(25.1 vs 23.6 us at 8 KiB): the cross-root hop is not the problem, the TP8 ring has 8 steps (115 us at 4 KiB vs 52 us for 4 ranks).

## Cross-root P2P is direct

Kernel stores and loads target the peer's BAR address (no host staging exists for them); cross-root push reaches the same 25 GB/s
as intra-switch and costs only 0.7 us more, which is a PCIe/IF hop and not a trip through host DRAM. KFD marks every pair P2P-capable and
RCCL logs `P2P/direct pointer` for the pairs that cross roots. The remaining cost is on the path itself: reads across roots are slow
(5.7 GB/s) and 3 or more concurrent distinct cross-root pairs collapse the posted-write throughput. [INFERENCE] both point at the
root-complex/IF forwarding of P2P TLPs (read completions, posted-write credits across several flows) and not at the PLX, since the
PLX alone sustains 105 GB/s inside a switch; no measurement here isolates which stage limits it.

## Proposals, ranked by evidence (tested 2026-11-07, verdicts in "Results of the tests")

1. **Collective design rules (no host change; measured).** Cross-root step: push, never pull (5.7 vs 25 GB/s); use one bidirectional
   pair per direction (53.5 GB/s total) or at most 2 concurrent pairs; never 3-4 independent pairs for payloads above ~64 KiB (aggregate
   falls to 17-18.7 GB/s). Fan-in/fan-out from a single GPU keeps 25.6-27.7. Intra-switch traffic can run in parallel with the cross step
   with no penalty. Below 64 KiB all concurrency levels cost 7-10 us. TP4xPP2: keep each TP4 group inside one switch; PP point-to-point
   traffic crosses roots at ~25 GB/s regardless of TP load.
2. **Make errors observable and recoverable (evidence: `_OSC` firmware-first, counters 0, history of fatal CmpltTO to sync flood).**
   Test `pcie_ports=native` so the kernel owns AER/DPC. Expected: real `aer_*` counters and kernel logs for the next fault; possibly a
   recoverable GPU reset instead of a whole-box sync flood [INFERENCE]. Risk: medium (unsupported by the platform's `_OSC`; the
   firmware may still raise SERR). Apply: `sudo grubby --update-kernel=ALL --args="pcie_ports=native"` and reboot. Rollback:
   `sudo grubby --update-kernel=ALL --remove-args="pcie_ports=native"` and reboot. BIOS alternative: look for an AMD CBS NBIO/RAS option that
   disables PCIe sync-flood generation or hands AER to the OS (exact names vary by BIOS; not read here).
3. **Test whether the IOMMU path causes the 3+-flow cross-root collapse [INFERENCE, untested].** `iommu=pt` still routes upstream
   P2P TLPs through the root-port ACS redirect and the IOMMU front end. Boot once with `iommu=off` (or BIOS IOMMU Disabled) and rerun
   `p2p_probe bw 0.4 1048576 128 p:0>4 p:1>5 p:2>6 p:3>7` (today 17.0 GB/s total). Expected: unknown; if it recovers to ~27 or more, the
   IOMMU is the cause. Risk: low for this box (no VFIO/VMs), interrupt remapping is lost. Apply: `sudo grubby --update-kernel=ALL --args="iommu=off"`
   and remove `iommu=pt amd_iommu=on`. Rollback: `sudo grubby --update-kernel=ALL --remove-args="iommu=off" --args="iommu=pt"` and reboot.
4. **Root-port ACS clear, runtime, instant test [low probability].** Root ports have ACS 0x001d. Peer-to-peer between roots passes
   through them upstream, so redirect should not matter; the test costs seconds. Apply: `sudo setpci -s 00:01.1 ECAP_ACS+6.w=0000` and
   `sudo setpci -s c0:03.1 ECAP_ACS+6.w=0000`, rerun the 3-pair probe. Rollback: `sudo setpci -s 00:01.1 ECAP_ACS+6.w=001d` and the same for
   c0:03.1 (or reboot). Risk: none for P2P among identity-mapped devices.
5. **Clean the command line (cosmetic).** `amd_iommu=on` is rejected by the kernel (`Unknown option - 'on'`) and does nothing. Remove it on
   the next planned reboot: `sudo grubby --update-kernel=ALL --remove-args="amd_iommu=on"`. Rollback: add it back.
6. **PLX switch firmware (no evidence; do not touch).** The firmware version is not visible from the OS here (management endpoints
   1c:00.0 and e0:00.0 need Broadcom tooling, not installed). Without an observed fault on the switch and with intra-switch P2P at 26 GB/s
   non-blocking, an upgrade has no measured upside and a brick risk. [UNOBSERVED]
7. **Not worth changing (evidence).** ARI, 10-bit tags (root complex lacks the completer capability), relaxed ordering (already on), MPS
   (GPUs cap at 256, already 256), ASPM (off), xGMI/IF link settings (single socket), GPU placement (all pairs within 3 %, intra and cross
   are equal for pushes; db as a destination is 8 % faster, not actionable).

## Results of the tests (2026-11-07, par1-cs25, kernel 7.1.7-100.p2p_post7)

Final decision: **remove `amd_iommu=on`, add `pcie_ports=native`, change nothing else.** Proposals 3 (`iommu=off`) and 4 (root-port ACS
clear) are refuted and not applied; proposal 5 is applied; proposal 2 is applied for observability only (it fixes no performance issue).

Final command line (persisted on the default BLS entry with `grubby`, plus `/etc/kernel/cmdline` and `GRUB_CMDLINE_LINUX_DEFAULT` so a
new kernel install does not bring `amd_iommu=on` back; verified after a reboot, `/proc/cmdline`):

```text
ro rd.lvm.lv=fedora/root rhgb quiet iomem=relaxed iommu=pt pcie_acs_override=downstream,multifunction amdgpu.ppfeaturemask=0xfff7ffff pci=realloc=off,disable_acs_redir=pci:1000:c010 pcie_ports=native
```

Method. Each state: reboot, health check (`health.sh`: 10 GPUs, 16 GT/s x16 on every V620/W7800 port, ACS, IOMMU mode, journal), the
same P2P subset (`tools/rdna/p2p_probe/subset.sh`: probe suite `sub` x3 plus RCCL TP8 all_reduce x2, production env, ~95 s), then one
recipe-native TP8 Flash-Next boot (`tools/rdna/tp8pp2/driver.sh CFG=rtp8 RECIPE=flashnext-tp8-mtp2`, tree `vllm-hier-9b96ecfa7a` =
rdna_extras `9b96ecfa7a`, production compiled mode, FULL_AND_PIECEWISE, probes, PPL probe, concurrency probe + 3 repeats, one 1k/512
c8 cell with 32 requests). Test entries were one-shot (`grub2-reboot`) so the default entry stayed the known-good original until the
last step. Cmdline states: **0** original (`amd_iommu=on iommu=pt`, no `pcie_ports`), **A** `amd_iommu=on` removed + `pcie_ports=native`,
**B** A with `iommu=pt` replaced by `iommu=off`, **F** final = A persisted.

### Step 1: root-port ACS clear (runtime, no reboot): no effect

`setpci -s 00:01.1 / c0:03.1 ECAP_ACS+6.w=0000` (0x001d -> 0), probe subset before and after, then restored to 0x001d.

| Metric (GB/s unless noted) | base | ACS 0 |
|---|---|---|
| 1 pair push 1 MiB | 25.16 | 25.17 |
| 2 pairs | 27.62 | 27.61 |
| 3 pairs | 18.67 | 18.76 |
| 4 pairs | 16.74 | 16.77 |
| 4+4 bidirectional | 35.40 | 35.39 |
| cross-root pull | 5.65 | 5.65 |
| `hipMemcpyPeerAsync` cross 64 KiB / 1 MiB | 4.59 / 19.97 | 4.60 / 19.97 |
| RCCL TP8 4 KiB / 8 MiB (us, two runs) | 113,110 / 592,590 | 110,109 / 592,593 |

Verdict: **refuted.** Identical within spread; the redirect bits on the root ports are not in the collapse or the pull penalty.

### Steps 2 and 3: pcie_ports=native (A) and iommu=off (B), P2P subset

Median of 3 repeats (probe) and per-run values (RCCL). States 0 = `s1_base`, A = `s2_A`, B = `s3_B`.

| Metric (GB/s unless noted) | 0 | A | B |
|---|---|---|---|
| 1 pair push 64 KiB / 1 MiB | 25.19 / 25.16 | 25.19 / 25.16 | 25.20 / 25.18 |
| cross-root pull 64 KiB / 1 MiB | 5.67 / 5.65 | 5.67 / 5.65 | 5.68 / 5.65 |
| intra-switch push / pull 1 MiB | 24.55 / 24.39 | 24.54 / 24.39 | 24.55 / 24.39 |
| `hipMemcpyPeerAsync` cross 64 KiB / 1 MiB | 4.59 / 19.97 | 4.59 / 19.97 | 4.60 / 19.98 |
| 2 concurrent cross-root pairs | 27.62 | 27.60 | 27.64 |
| **3 concurrent cross-root pairs** | 18.67 | 18.41 | 18.57 |
| **4 concurrent cross-root pairs** | 16.74 | 16.76 | 16.76 |
| 4+4 bidirectional | 35.40 | 35.36 | 35.38 |
| intra-switch ring (4 pairs) | 105.08 | 105.09 | 105.06 |
| one-way flag latency 4 / 16 / 64 KiB x4 pairs (us) | 6.74 / 7.11 / 10.33 | 6.82 / 7.29 / 10.27 | 6.76 / 6.93 / 10.24 |
| flat 8-GPU all-to-all step 4 / 16 / 64 KiB (us) | 14.8 / 20.9 / 62.8 | 15.2 / 20.9 / 62.7 | 14.0 / 20.6 / 62.6 |
| RCCL TP8 4 KiB (us) | 113, 110 | 111, 109 | 110, 111 |
| RCCL TP8 64 KiB (us) | 181, 176 | 186, 186 | 182, 186 |
| RCCL TP8 1 MiB (us) | 204, 181 | 179, 177 | 177, 199 |
| RCCL TP8 8 MiB (us) / busBW | 592, 590 / 24.8 | 592, 592 / 24.8 | 592, 590 / 24.8 |

Verdicts:

- `iommu=off` (B): **refuted.** IOMMU hardware is really off (no `/sys/class/iommu`, 0 IOMMU groups, no AMD-Vi lines). The 3-pair
  collapse (18.6 vs 18.7), the 4-pair collapse (16.8 vs 16.7), the 5.65 GB/s cross-root pull and the 4.6 GB/s `hipMemcpyPeerAsync`
  64 KiB are unchanged. The IOMMU front end is not the cause. Reasoning about the other options: `pcie_acs_override` and
  `disable_acs_redir=` are PCI-core/quirk code that acts on the ACS registers and does not need an IOMMU (ACS stayed 0x1d on the root
  ports and 0x0000 on the PLX ports in B, `health.sh`); with the IOMMU off the groups just disappear. Nothing else on the line depends
  on it. B costs interrupt remapping and has no measured gain, so it is not kept.
- `pcie_ports=native` (A): **P2P and RCCL identical.** What it does: the kernel now owns AER and DPC on the 17 root ports (`AER: enabled
  with IRQ ..`, `DPC: enabled with IRQ ..`, error containment capabilities RPExt+ PoisonedTLP+ DL_ActiveErr+) although `_OSC` still says
  the platform does not support AER/DPC/LTR. The `aer_*` sysfs counters are live (state B/F: a few ports show `NonFatalErr` 1-2, root
  ports `aer_rootport_total_err_cor` 10).
- The cause of the 3+-pair cross-root collapse and of the 5.65 GB/s pull is therefore neither root-port ACS nor the IOMMU
  [INFERENCE: the data fabric / IO-die routing between the two root complexes, outside anything the OS controls].

### What pcie_ports=native logs at boot (all boots with it: A, B, F)

Within about 2 s of boot (before any workload): 36-45 AER/GHES lines per boot, then nothing. The records are the sticky status bits left
by firmware/POST, now reported once by the kernel and cleared:

- Correctable `aer_cor_status 0x00002000` (bit 13, Advisory Non-Fatal) on PLX ports and on the V620/W7800 functions
  (`c0:03.1 <- c6:0c.0`, `00:01.1 <- 02:0c.0`, `c5:00.0`, `c6:04.0`, `c6:08.0`, `c7:00.0` ...), mask 0.
- GHES records with `aer_uncor_status 0x00100000` = Unsupported Request, mask 0x00100000 (masked), on V620 ce/d3/d8/db and the two
  W7800 functions (07:00.x, cb:00.x), followed by `AER: buffer overflow in recovery for 0000:ce:00.0 ..` (the AER recovery queue
  overflowed because 6-10 records arrived in the same millisecond; cosmetic).
- No `severity=Fatal`, no `Uncorrected` event, no DPC trigger, no link reset, no IOMMU fault. Zero AER/DPC/GHES lines afterwards, through
  four TP8 boots of about 8 minutes each and the P2P/RCCL runs.

Not measured: whether native DPC/AER recovery turns a real fatal fault (the 2026-11-05 completion timeout) into a recoverable GPU reset
instead of a sync flood. No fault injection was done; this stays [INFERENCE]. If a real fault appears, kernel counters and dmesg now
show its device and layer.

BMC side (read-only observation, `ipmitool sel list`): the SEL logs `Critical Interrupt | PCI PERR | Asserted` in batches at every
reboot, 12-52 entries at shutdown and 20-22 at POST, for the original cmdline's shutdown (01:40, 02:22) as well as for the native ones
(02:11, 02:24); none during any run, and one `Memory Correctable ECC` at 01:42:22. So the PERR batches do not depend on
`pcie_ports=native`. The SEL had been empty before 01:40 on 2026-11-07, so older occurrences are [UNOBSERVED].

### TP8 Flash-Next smoke per state (`flashnext-tp8-mtp2`, rdna_extras `9b96ecfa7a`, production compiled mode)

| State | READY | Probes | PPL | Concurrency + 3 repeats | 1k c8 ITL p50 / TTFT mean / output | New dmesg AER/AMD-Vi/amdgpu/KFD lines |
|---|---|---|---|---|---|---|
| 0 original | 241 s | 3/3 | 4.80 | PASS, 3/3 | 37.4 ms / 0.81 s / 319 tok/s | 0 |
| A native | 351 s | 3/3 | 4.81 | PASS, 3/3 | 37.4 ms / 0.77 s / 328 tok/s | 0 |
| B native + `iommu=off` | 240 s | 3/3 | 4.83 | PASS, 3/3 | 37.2 ms / 0.79 s / 308 tok/s | 0 |
| F final (persisted A) | 250 s | 3/3 | 4.85 | PASS, 3/3 | 37.2 ms / 0.78 s / 310 tok/s | 0 |

All four are equal within run-to-run spread (output 308-328 tok/s = 6 %, one cell each; ITL p50 37.2-37.4 ms). The 351 s READY of A is
one sample after a reboot with the page cache cold and docker still starting [INFERENCE]; the other three are 240-250 s. This is a
smoke, not an A/B performance measurement. The doc `tp8-pp2-flashnext-2026-11-06.md` has the matched cells for the hier tree.

### Rollback

```bash
# back to the exact original command line (one reboot):
sudo grubby --update-kernel=DEFAULT --remove-args="pcie_ports=native" --args="amd_iommu=on"
# /etc/kernel/cmdline and /etc/default/grub were edited too; originals:
sudo cp -a /root/kernel_cmdline_backup_2026-11-07 /etc/kernel/cmdline
sudo cp -a /root/etc_default_grub_backup_2026-11-07 /etc/default/grub
# BLS entry files before any edit: /root/bls_entries_backup_2026-11-07
# only drop native AER/DPC (keep the cleaned line):
sudo grubby --update-kernel=DEFAULT --remove-args="pcie_ports=native"
# one-shot test of another line without touching the default (falls back on a failed boot):
sudo grubby --add-kernel=/boot/vmlinuz-7.1.7-100.p2p_post7.fc43.x86_64 --initrd=/boot/initramfs-7.1.7-100.p2p_post7.fc43.x86_64.img \
  --copy-default --title=par-test --args="..."; sudo grub2-reboot <entry id>; sudo systemctl reboot
```

Out of band: the BMC (192.168.1.82, ipmitool lanplus; credentials in the box owner's hands) answers `chassis power status` and
`sol activate` without the OS; used as the recovery path for these reboots, never needed (the box was back on ssh in 96-190 s every time).
Risk that remains with `pcie_ports=native`: a platform that says no to AER/DPC in `_OSC` may still raise firmware SERR/sync flood on a
real fatal error; the kernel path adds an extra handler, it does not remove the firmware one [INFERENCE].

## Open items

- Cause of the 3+ concurrent-pair cross-root collapse and of the 5.7 GB/s cross-root read rate: root-port ACS (step 1) and the IOMMU
  (`iommu=off`, step 3) are ruled out; what remains is the data fabric between the two root complexes (not controllable from the OS)
  and the PLX upstream links' arbitration [INFERENCE]. The design rules in proposal 1 already avoid both.
- Whether native DPC/AER recovery contains a real fatal fault: not testable without injection.
- Whether the W7800 runtime-PM wake (`Configured ACS to 0x0000`) can disturb a running TP8 server: not exercised; no workload opens the W7800, only `rocm-smi`-style queries do.
- RCCL numbers used the system ROCm RCCL and rccl-tests, not the vLLM wheel's RCCL; Tp8Pp2 measures the vLLM path with
  `benchmarks/kernels/benchmark_rdna_allreduce.py --world 8`.
- 4-pair latency numbers are a mean over 2000 back-to-back iterations (3 repeats), not a per-op median.
