# rdna_ars: reduce-scatter + allgather all-reduce (plan for a future PR)

Status: deferred (future PR). Code saved on branch
`wip/rdna-ars-snapshot-2026-10-06` (opengfx1030/vllm-rdna, commit `dcc3498fd`, on top of
0.28 `rdna_extras` @ `c6d99ca30`). The snapshot holds the working tree as it was on
2026-10-06, so it is not lost.

## Why

The one-shot `rdna_ar` (in use for payloads up to 64 KiB) pushes the full input to every
peer: `(W-1)·N` bytes per rank. RCCL's ring moves `2(W-1)/W·N`. Above 64 KiB (prefill,
large decode batches, TP=8) RCCL wins. The push-all `rdna_ar2` was correct, but it ran at
0.73× / 0.46× / 0.20× / 0.09× RCCL at 2.6 / 5.2 / 10.5 / 21 MB
(`docs/rdna2/tp4-allreduce-ceiling-2026-10-06.md`). A reduce-scatter + allgather shape
moves the same bytes as the ring. It is the only custom route that could beat RCCL above
64 KiB, and it matters most for TP=8 across both PLX switches.

## What the snapshot contains

| File | What |
|---|---|
| `csrc/rocm/rdna_ar2_rs.cu/.cuh` | `rdna_ars` reduce-scatter + allgather kernel; world size 2..8; no grid-wide barrier |
| `csrc/rocm/rdna_allreduce2.cu/.cuh`, `rdna_ar2_v2.cuh` | push-all `rdna_ar2` (correct, loses to RCCL); shared transport (P2P or host, chosen with `hipDeviceCanAccessPeer`) |
| `benchmarks/kernels/ra2rs_protocol_model.py`, `ra2_protocol_model.py`, `ra2_differential.py` | protocol models (28/28) and a differential test harness |
| `csrc/rocm/rdna_allreduce.cu/.cuh`, `ops.h`, `torch_bindings.cpp`, `rdna_all_reduce.py` | two-shot removal (ported separately to v0.31) |

Evidence: protocol model 28/28; 4-rank hardware correctness 4/4 `ok=True`
(`docs/rdna2/par1-cs25-reset-cause-2026-10-06.md`). **Latency vs RCCL never measured.**

## Plan

1. **Port** the `rdna_ars` kernel + transport into `csrc/rocm/rdna/allreduce/` (new layout).
   Register it as `_rocm_C::rdna_ars_*` next to the one-shot, behind an opt-in
   (`VLLM_RDNA_AR_ALGO=rs`), off by default.
2. **Correctness:**
   - port the protocol model and the differential test into `tests/distributed/`
     (fp16/bf16, sizes 64 KiB … 64 MiB, odd tails, W = 2 / 4 / 8);
   - add a boot self-test like the one-shot's, which falls back to RCCL on mismatch;
   - add a capture/replay test, since it must work under FULL_AND_PIECEWISE: current
     stream, no host sync, fresh outputs, no shared persistent output buffer.
3. **Latency A/B vs PYNCCL** at 64 KiB, 256 KiB, 1, 2.6, 5.2, 10.5 and 21 MB:
   - TP=4 on one PLX;
   - TP=8 across both PLX switches.
   Land it only for the size range where it wins, as a gate like `VLLM_RDNA_AR_MAX_KB`.
4. **End-to-end:** Flash-Next and 27B AWQ, 16k/1k and 1k/512 at c=1/c=8 (prefill, decode,
   TTFT, ITL), TP=4 and TP=8; probes + PPL + prefix.
5. **Hardware safety:**
   - the chassis has logged PCI SERR/PERR, and the stock custom AR SERRs it;
   - check the BMC SEL (`ipmitool sel list`) and dmesg before and after every run;
   - stop at the first bus error, and keep RCCL as the default until a run is clean.
6. **Optional later:** int8 transport (halves the bytes; int32 accumulate with a
   saturating cast, reconciled scales).

## Known failure mode to avoid

The old two-shot returned zeros above the one-shot gate (a contiguous zero run inside one
peer's slice). Only block 0 waited on the peer flags; fixing that cleaned 8194 B but larger
sizes stayed broken, and six hypotheses were falsified. Use per-block flags, cover every
slice in the tests, and keep the differential harness in CI.
