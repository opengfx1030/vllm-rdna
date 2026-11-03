# Two-lane engine: acceptance verdicts

Status: **in progress**. Branch `feat/two-lane`; contract in
`~/w4a8_runs/lanes/async/two-lane-contract.md` (untracked copy in the tree).

The engine is opt-in (`TWO_LANE=1`); the flag-off path stays the rdna_extras
default. Flag-on recipes are `tools/rdna/recipes/flashnext-mtp2-twolane.env`
(TP4) and `tools/rdna/recipes/27b-awq-twolane.env` (TP2).

## M1 TP1 worker driver

Tool: `tools/rdna/prof/lane_driver.py`, Qwen3.5-0.8B fp16, one V620 (HIP 9),
TP1, kv 24 GiB, `RDNA`/`ROCM` attention as the serving path. Reports:
`~/w4a8_runs/lanes/async/accept/m1*/lane_driver.json`.

| gate | result | numbers |
|---|---|---|
| (a) handoff L in {1, 5, 1023, 1024, 1025, 2047, 2048, 4097, 16384, 16385} | **PASS** (eager) | all lengths byte-equal; 16384/16385 included with `max_model_len=32768` |
| (b) E-GIL D p99 / D p50 solo | **FAIL (marginal)** | 1.529 (gate <=1.5); D solo p50 11.17 ms, D concurrent p99 17.1 ms. Earlier run 1.489. P eager Python launch is the residual GIL cost |
| (c) P chunk rate >= 45% of solo | **PASS** | 98.0% |
| (d) 10 min, no hang/error, strict clean | **PASS** | 309 soak iterations, 0 mismatches, 0 errors, `d_reqs=0`, `p_reqs=0` |
| (e) KFD census 7 | **PASS eager** | 7 queues; 8 with D graphs (`GPU_MAX_HW_QUEUES=1`), at the hard limit and not the gate |

Findings:

- The L<=16 mismatch was the driver handing out **block id 0**, vLLM's null
  block, not an S5/S6 bug. The driver now reserves block 0 (real
  `KVCacheManager` never allocates it); all lengths match.
- The `post_update_kernel` GPU fault at 16k was the driver's default
  `max_model_len=16384`: a 16385-token request needs 1025 block-table slots,
  the buffer had 1024. Default is now 32768.
- **Open:** with D compiled (`--compile`), P's first token matches the baseline
  but D's first decode diverges for every handoff length; eager D is exact.
  GDN state arenas are now disabled in lane mode and the divergence remains, so
  the captured handout path (input/state buffer of a NEW request) is the
  suspect. This blocks M2/M4 handoff gates under the production compile mode.
- At TP1 the graph build needs `GPU_MAX_HW_QUEUES=1` to stay within 8 queues;
  the production recipes pin 2. The 8th queue with graphs must be removed
  before M4 (gate 7).

## M2 engine at TP1

Pending the compiled handoff fix. `two_lane_gate.sh` is in place; the real
server smoke is running with the 27B recipe overridden to TP1/Qwen3.5-0.8B.

## M3 TP2 27B / M4 TP4 Flash-Next

Not started; G0-G7, G4 split table (16/18/20/24, including 16/20 and 20/16)
and the keep/drop recommendation follow after M2.
