# TP=4 all-reduce at serving sizes (2026-10-10)

Box `par1-cs25`, 4x V620 on HIP 2-5 (PLX-B, buses 47/4A/52/55), tree
`rdna_extra/v0.31.0` + agent U commits. Tool:
`benchmarks/kernels/benchmark_rdna_allreduce.py` (vLLM's own TP group, so the
backends are the ones serving uses; every result is checked against the exact
sum), sweeps by `tools/rdna/port_v031/ar_sweep.sh`. Times are microseconds per
call, max over ranks of the median of 7 repeats x 20 calls. "graph" replays the
calls from a captured CUDA graph (the FULL decode path). Raw results are in
`~/w4a8_runs/port-v031/u-arbench/*.json` on the box.

## Topology (NCCL_DEBUG=INFO)

RCCL builds 2 channels, ring `0 1 2 3` = buses 47 -> 4A -> 52 -> 55 -> 47, all
links `P2P/IPC`. That follows the sub-port pairs (47,4A) and (52,55): two hops
inside a sub-port pair and two across. RCCL warns `graphUsageMode is set to 0 but
the user is capturing graphs` for every capture. It is harmless here, but small
messages are ~10-15 us slower inside graphs than eager (below).

## RCCL (pynccl), eager

| tokens x hidden | KiB | default | NCCL_PROTO=LL | LL128 | NCCL_ALGO=Tree | NCCL_MIN_NCHANNELS=8 |
|---|---:|---:|---:|---:|---:|---:|
| 1 x 2048 | 4 | 62.7 | 48.1 | 53.8 | 59.8 | 65.1 |
| 8 x 2048 | 32 | 57.5 | 51.0 | 57.1 | 73.9 | 83.4 |
| 16 x 2048 | 64 | 59.7 | 53.1 | 59.3 | 87.9 | 63.5 |
| 24 x 2048 | 96 | 62.8 | 56.6 | 61.0 | 88.8 | 64.3 |
| 64 x 2048 | 256 | 83.1 | 93.6 | 72.9 | 94.0 | 76.3 |
| 256 x 2048 | 1024 | 101.2 | 338.1 | 101.7 | 172.9 | 128.8 |
| 1024 x 2048 | 4096 | 274.1 | 1314.0 | 270.6 | 457.3 | 265.8 |
| 2048 x 2048 | 8192 | 500.7 | 2612.4 | 499.7 | 782.3 | 498.2 |
| 8 x 5120 | 80 | 61.4 | 53.3 | 60.7 | 84.0 | 64.4 |
| 24 x 5120 | 240 | 97.8 | 91.2 | 72.5 | 96.6 | 72.5 |
| 256 x 5120 | 2560 | 185.3 | 831.3 | 184.2 | 314.4 | 185.0 |
| 1024 x 5120 | 10240 | 615.7 | 3259.7 | 616.2 | 950.4 | 600.8 |
| 2048 x 5120 | 20480 | 1227.7 | 6607.2 | 1224.8 | 1757.1 | 1171.5 |

- **Prefill sizes are at the link limit.** At 4-20 MB the ring reaches 23-25.6
  GB/s bus bandwidth, the practical PCIe Gen4 x16 ceiling. No RCCL setting or
  custom all-reduce shape can make these faster; only fewer bytes (int8
  transport, fewer all-reduces) or overlap can. (The 2026-10-06 report put RCCL
  at 0.06 ms for 10-21 MB; that is not physically possible and was an unsynced
  timing.)
- The default already chooses Simple (`NCCL_PROTO=Simple` is identical). LL wins
  5-10 us below ~100 KiB and is 3-5x slower above. LL128 equals Simple. Tree is
  worse everywhere.
- `NCCL_MIN_NCHANNELS=8`: librccl prints "NCCL_MIN_NCHANNELS set by environment
  is ignored due to less than 8 GPUs". The -4.6 % at 20 MB is therefore noise or
  a different channel pick and is not landed.
- Floor: ~50-60 us eager, 64-76 us in a graph, for anything up to ~100 KiB.

## rdna_ar one-shot vs RCCL (gate raised to 1 MiB for the test)

| tokens x hidden | KiB | one-shot eager | RCCL eager | one-shot graph | RCCL graph |
|---|---:|---:|---:|---:|---:|
| 8 x 2048 | 32 | 31.8 | 57.5 | 27.4 | 75.6 |
| 16 x 2048 | 64 | 41.6 | 59.7 | 36.9 | 66.8 |
| 24 x 2048 | 96 | 55.1 | 62.8 | 49.1 | 69.4 |
| 8 x 5120 | 80 | 47.3 | 61.4 | 45.1 | 64.5 |
| 16 x 5120 | 160 | 68.1 | 75.7 | 65.5 | 76.1 |
| 24 x 5120 | 240 | 87.6 | 97.8 | 79.3 | 78.8 |
| 64 x 2048 | 256 | 99.6 | 83.1 | 83.6 | 80.9 |
| 128 x 2048 | 512 | 149.8 | 81.3 | 148.7 | 77.9 |
| 256 x 2048 | 1024 | 264.6 | 101.2 | 275.6 | 94.6 |

The one-shot beats RCCL up to ~160 KiB (by 15-20 us per call at 80-96 KiB, the
27B c=8 decode and Flash-Next MTP=2 c=8 verify sizes). They are even at
240 KiB, and RCCL wins from 256 KiB. The 64 KiB gate leaves those decode sizes on
RCCL. Raising it to 128-160 KiB is the candidate change; it still needs the
serve A/B (27B AWQ and Flash-Next MTP=2 c=8 decode, stall probe) and a
fine-grained sweep at 64-256 KiB. The BMC showed no new PCI SERR/PERR across the
sweep. The newest is at 04:19, before these runs; the SEL listing on this BMC
drops entries, so `ar_sweep.sh` compares the newest PCI record, not a count.

Mixed sizes (256 KiB-2 MiB) sit at 80-185 us, 30-60 % of the link. A
reduce-scatter + allgather custom kernel (`rdna_ars`) could only win there.
In a 256-token mixed step that is ~1 % of step time, so it is not worth the risk
on this chassis.
