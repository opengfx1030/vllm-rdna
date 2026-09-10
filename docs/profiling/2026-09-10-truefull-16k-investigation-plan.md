# 16k TRUE FULL duct — investigation plan (2026-09-10)

Status: 16k/1k c=1 deterministic `Parisduct` + permanent graph poison (seq-after 0/3)
on every config tried. 1k/512 through c=8 and sequential greedy are green when
isolation holds. This plan supersedes serve-by-serve guessing: each phase names a
mechanism or eliminates it with a discriminating experiment.

## Evidence-locked (do not re-litigate)

| # | Fact | Where proven |
|---|------|--------------|
| 1 | 16k c=1 → first token correct, rest `duct`, then ALL later greedy `duct` | serve26/29/33/34/35 |
| 2 | 1k/512 c=1/4/8 + seq-after green | serve26/29/33 (8/8) |
| 3 | Prefill logits fine; decode REPLAY is what breaks; stays broken | first token + seq-after |
| 4 | Zero torch-allocator events in graph pool (0,1) during 16k — incl. transient, with history recorder ON during the failing run | poolwatch baseline byte-exact (65837056) on serve29/33/34; serve35 replay dump: 1206 blocks, all pre-capture |
| 5 | No persist grows during 16k; persist slot selection provably eager post-freeze (`g_rdna2_capture_frozen` in binary; freeze log fires) | serve29/33 logs, nm -D |
| 6 | Graph pool (0,1) has ~1 MiB free post-capture; filling free blocks does NOT prevent duct | serve35 `[fill] free=1 MiB; occupied 128 MiB` |
| 7 | `expandable_segments:False` after capture does NOT fix 16k and REGRESSES 1k c=8 to 4/8 (now env-gated `VLLM_ROCM_DISABLE_EXPANDABLE_AFTER_CAPTURE`, default off) | serve32 vs serve33 |
| 8 | NOT: KV capacity (22% at 16k c=1), FA/W4A16 dispatch, HIP capture false-positive (raw HIP False + frozen flag), history-recorder timing (serve34/35 fail with recorder on; serve30 "PASS" had `n_prompt=0` = false pass) | prior serves |
| 9 | Runner is V2 (`VLLM_USE_V2_MODEL_RUNNER=1` in serve script; "V1 LLM engine" log = engine generation, not runner) | serve_gfx1030_full.sh:27 |
| 10 | Persist-pin rebuild landed (frozen-gate, d0>=256) — .so 71434648 @ 11:22; no 128 MiB floor anywhere | so_rebuild_proof, serve30+ |

## The one new lead

`causal_conv1d` PREFILL is still Triton/FLA (HIP port `causal_conv1d_fwd_rdna2.cu`
exists but its FIR formula is wrong and dispatch is gated off — see AGENTS.md).
1k prompts exercise it at 1024-token shapes; 16k exercises it at 1568/704-token
chunks and 16k seqlen buckets — shapes never compiled during warmup, so Triton
JIT/autotune runs DURING live traffic inside a process holding HIP graphs. That
failure class has broken graph state on gfx1030 before (709 context-destroyed;
`_causal_conv1d_fwd_kernel` capture fault). Fits all 10 facts above.

## Hypotheses (ranked)

- **H1 — Triton JIT/autotune during live 16k chunks poisons HIP graph state.**
  Trigger: new conv1d shapes at 1568/704. (Leading.)
- **H2 — multi-chunk iteration (11 chunks vs 1) corrupts shared state the first
  decode replay depends on** (per-chunk metadata, GDN state handoff, hybrid
  2-group page accounting), independent of shapes.
- **H3 — stale-pointer write via non-torch memory** (hipMalloc'd immortals,
  Triton device scratch outside the torch allocator, RCCL). Fallback if H1/H2
  come up empty.

## Phases

### Phase 1 — characterize the trigger (CLI-only, ~30 min, one launch)

serve36 = serve33 config (frozen-gate .so, expandable ON, ws pretouch, cheap
poolwatch) + `VLLM_LOG_GDN_DISPATCH=1`.

1. Ascending prefill-length sweep at c=1, output-len 64, stop at first FAIL:
   1024 (control, 1 chunk) → 2048 (1 chunk) → 2049 (2 chunks: 1568+481) →
   4096 (3) → 8192 (6) → 16384 (11).
   - FAIL at 2049 → multi-chunk iteration is the trigger (H2).
   - FAIL only at some longer length → length/count-dependent; the threshold
     localizes the mechanism (pages, buckets, or a specific shape).
2. Relaunch with `--max-num-batched-tokens 32768` and run 16384 as ONE
   skip-compiled chunk.
   - Green → chunk-iteration (H2), not the prefill itself.
   - `duct` → the 16k prefill content itself (H1 shapes / H3 length).
3. Verify GDN prefill dispatch is HIP on this build (3648-style
   `[gdn-dispatch] ... PREFILL -> HIP chain` lines). If any layer falls to
   Triton at 16k buckets, that is a second live-JIT source.

### Phase 2 — kill the Triton in the 16k path (H1)

2a. **Pre-warm (cheap, Python-only)**: during post-capture warmup, call the
    Triton conv1d fwd at the exact skip-compiled shapes (1568 and 704 token
    chunks, long seqlens) under isolation, so no JIT/autotune can fire live.
    Re-run 16k c=1. Green ⇒ live-JIT was the trigger.
2b. **Full HIP (the real fix, aligns with project goal)**: fix the
    `causal_conv1d_fwd_rdna2` FIR formula (`state[k] * w[state_len-1-k]`,
    state has `state_len` entries not `width`; same fix for
    `causal_conv1d_update_rdna2`), pass the standalone correctness probe
    (`/tmp/test_causal_conv1d_fwd_rdna2.py` pattern from AGENTS.md), enable
    both dispatches, rebuild, re-run 16k. This removes the last Triton kernel
    from the prefill path entirely.

### Phase 3 — if still duct: rocprof VA-filtered kernel trace (H3)

Use ROCm 7.14 rocprof (NOT system 7.2.0 — see rdna-kernel-debug skill).
Graph-pool VA ranges are known from the serve35 replay dump. Trace a 16k
prefill + first decode replay; any kernel writing into (0,1) VA during
skip-compiled steps is the culprit. Also checksum candidate graph-resident
buffers (persist capture slots via a small C++ accessor, GDN/FA capture tables,
static runner input buffers, `_full_capture_keepalives`) before/after the 16k
prefill to name the victim tensor.

### Phase 4 — fix + full regression matrix

Sequential 3/3 → 1k/512 c=1/4/8 + seq-after → 16k/1k c=1 + seq-after →
16k c=4 + seq-after → 16k c=8. All must pass on one server, then reproduce on a
second fresh launch. Kill servers by PID only.

## Current experiment stack (keep using)

- `VLLM_RDNA_POOLWATCH=1` — byte-watch on (0,1), zero overhead.
- `VLLM_RDNA_POOLWATCH_HISTORY=1` + `VLLM_RDNA_POOLWATCH_DUMP_ON_REPLAY=1` —
  recorder + first-replay snapshot (decode drops to ~0.7 tok/s; debug only).
- `VLLM_RDNA_FILL_GRAPH_POOL=1` — occupies (0,1) free blocks (diagnostic; not a
  fix — serve35 proved that).
- `VLLM_ROCM_DISABLE_EXPANDABLE_AFTER_CAPTURE` — env-gated OFF (regresses 1k).
- Isolation unbind (0,1)/(0,2)/(0,3) → beginAllocate(0,3) — KEEP (serve26).
- Frozen-gate 2048-row persist pin — KEEP; never re-introduce a byte floor.
