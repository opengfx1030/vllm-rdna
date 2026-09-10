# TRUE FULL gfx1030 — investigation journal (2026-09-10)

Model: Qwen3.8-27B-AWQ-INT4 (cyankiwi 63768c10), TP=2, .176 Radeon PRO V620,
venv-7.14.0, branch `rdna_extras` in `opengfx1030_vllm-rdna`.
Goal: TRUE FULL HIP graphs green for sequential, 1k/512 and 16k/1k through c=8
with prefix caching + mixed batch. Handover: `2026-09-10-truefull-handover.md`.
Plan: `2026-09-10-truefull-16k-investigation-plan.md`.

## Legend

Greedy probe = `tools/probe_greedy_correctness.py` (Paris / 1+1= / Berlin).
Bench = `tools/bench_streaming_ttft.py`. `duct` = repeated "duct" garbage token.
seq-after = greedy probe run after a concurrent/16k burst (graph health).

## Progression

| Serve | Config delta vs previous | 1k gate | 16k c=1 | Learning |
|-------|--------------------------|---------|---------|----------|
| 20–25 | GDN/FA pretouch, side stream, 10e9→6e9 KV pin (see handover) | mixed | Parisduct | pretouch necessary, not sufficient; 10e9 OOMs; 6e9 kept |
| 26 | **unbind graph pool (0,1) in isolation** | **8/8 + seq-after 3/3** | Parisduct + seq-after 0/3 | pool bind was a real leak for ~1024-token mixed; KEEP |
| 27 | 128 MiB persist floor + expandable:False | **3/8 REGRESSION** | not run | byte floors make leaked allocs deadlier; reverted |
| 28a | 2048-row persist pin (old gate: fires during capture, d0>0) | **0/8** (384 MiB splitk partial pinned into graph during piecewise capture) | — | pin must not fire during capture |
| 28b/29 | same .so (10:58) | **8/8 + seq-after 3/3** | **Parisduct + 0/3** | serve26 restored; 16k still deterministic; no persist grows during 16k; no bind failures |
| 30 | frozen-gate pin rebuild (11:22 .so) + history recorder | (not rerun) | "PASS" **but n_prompt=0 → false pass, never exercised 16k** | recorder tax = 0.65 tok/s decode; treat PASS as invalid |
| 31 | + expandable OFF after capture + ws pretouch 2048×5120/17408 | seq **0/3 duct at startup**; 1k c=1 PASS 13s later | — | startup duct was TRANSIENT (not reproduced on relaunch) |
| 32 | serve31 config relaunched | seq 3/3 ×2; c=1 1/1; c=4 4/4; **c=8 4/8** + seq-after 0/3 | — | expandable OFF regresses 1k c=8; poolwatch clean |
| 33 | expandable back ON (env-gated), ws pretouch kept | **8/8 + seq-after 3/3** | **Parisduct + 0/3** | bisect DONE: expandable-off was the serve32 regression; ws pretouch exonerated |
| 34 | + history recorder ON | — | **Parisduct** (recorder does NOT mask) | serve30's "PASS" was the n_prompt=0 anomaly, not the recorder |
| 35 | + fill (0,1) free blocks (only 1 MiB free; occupied 128 MiB) + recorder + dump-on-replay | — | **Parisduct**; replay dump: 1206 blocks all pre-capture, **zero torch-allocator events in (0,1)** | allocator-escape class DEAD (byte-watch + recorder + fill all negative) |

## Evidence-locked conclusions

1. Poison is graph-level and permanent; prefill logits always fine (first token
   correct); weights intact.
2. No torch-allocator involvement in pool (0,1) during 16k — persistent or
   transient — proven with the recorder on during failing runs.
3. Not KV capacity (22%), not FA/W4A16 dispatch, not HIP capture false-positive
   (frozen flag + guard live), not expandable remapping, not persist grows.
4. Runner is V2 (`VLLM_USE_V2_MODEL_RUNNER=1`); "V1 LLM engine" log = engine
   generation.
5. Remaining suspect class: **live Triton JIT/autotune during 16k chunks** —
   causal_conv1d prefill is still Triton/FLA (HIP port FIR bug, gated off), and
   1568/704-token chunks + 16k seqlen buckets are shapes warmup never compiled.
   gfx1030 has twice broken graph state from Triton-in-live-traffic (709,
   conv1d capture fault). Fits all evidence.

## Current experiment stack

- `VLLM_RDNA_POOLWATCH=1` byte-watch (free); `..._HISTORY=1` recorder (debug,
  ~0.7 tok/s); `..._DUMP_ON_REPLAY=1` first-replay snapshot.
- `VLLM_RDNA_FILL_GRAPH_POOL=1` (diagnostic; not a fix).
- `VLLM_ROCM_DISABLE_EXPANDABLE_AFTER_CAPTURE` env-gated OFF (regresses 1k).
- Isolation unbind (0,1)/(0,2)/(0,3) → beginAllocate(0,3) — KEEP.
- Frozen-gate 2048-row persist pin (.so 11:22) — KEEP; no byte floors ever.

## Next (parallel across 4 GPUs)

- GPUs 0,1 (:18094): Phase-1 trigger characterization — prefill-length sweep
  1024→2048→2049→4096→8192→16384 (min failing length) + single-chunk 16k
  (`--max-num-batched-tokens 32768`) to split "many chunks" vs "long prefill".
- GPUs 2,3 (:18095): fix `causal_conv1d_fwd_rdna2` + `causal_conv1d_update_rdna2`
  FIR pairing (`state[k] * w[state_len-1-k]`, state has state_len entries),
  standalone-verify on one GPU, rebuild, enable HIP dispatch, run 16k c=1.
  Completes the full-HIP prefill path AND tests the H1 hypothesis.
