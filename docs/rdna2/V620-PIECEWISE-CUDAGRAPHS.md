# PIECEWISE CUDA graphs on the V620 stack — 2026-09-23

Compiled piecewise CUDA graphs (`--compilation-config
'{"mode":3,"cudagraph_mode":"FULL_AND_PIECEWISE",...}'`) now boot, capture, and
serve correctly on the PR #17 stack (Qwen3.8-Flash-Next W4A16 AutoRound, TP4/EP4,
MTP2, PLE CPU offload, prefix caching, 4x V620 / gfx1030).

Everything below was measured on this box with
`ROCM_LIB=/opt/rocm/core-10.0/lib TUNEOP=1 PRELOAD=0` and the PR #17 TunableOp
table, using `<wsantos-home>/work/pw-serve.sh` (a copy of
`tools/rdna2/serve_v620_baseline.sh` with the compilation config exposed) and
`pr17-bench.py` (16k prompt, 1024 output tokens).

## Why it was not working

1. **`cudagraph_mode: PIECEWISE` with `mode: 0` silently produced no graphs.**
   `VllmConfig.__post_init__` overrode the mode to `NONE` because piecewise
   capture needs either Inductor splitting (`VLLM_COMPILE`) or breakable CUDA
   graphs:

   ```
   INFO [vllm.py:1500] Cudagraph mode PIECEWISE is not compatible with
   compilation mode 0. Overriding to NONE.
   WARNING [model_runner.py:930] Skipping encoder and decoder CUDA graph capture.
   ```

   The service still answered correctly, but every step ran eagerly.

2. **Compiled piecewise capture aborted on a hyperconnection stride assert.**
   With `mode: 3` the first capture failed with

   ```
   AssertionError: expected size 12==12, stride 4==336 at dim=0
   ```

   The HC injection is a slice of the fused `rdna_hc_mix` output:
   `dai[:, lora_rank : lora_rank + hc_count]` is a `(num_tokens, 4)` view with
   row stride `lora_rank + hc_count + pad_size = 336`. Inductor specializes on
   the trace-time stride, so the capture aborted whenever the runtime tensor was
   contiguous.

3. **MTP speculator graph dispatch did not follow the ROCm FULL→piecewise
   redirect.** With `FULL_AND_PIECEWISE` on ROCm, `rocm_full_executes_as_piecewise`
   makes the target model replay its piecewise graphs instead of capturing FULL
   ones. Two speculator paths still assumed a FULL graph existed:

   - draft prefill called `prefill_cudagraph_manager.run_fullgraph()` →
     `No cudagraph for BatchExecutionDescriptor(cg_mode=FULL, num_tokens=12,
     num_reqs=4, uniform_token_count=3)`;
   - the draft-decode manager (`FULL_DECODE_ONLY`, no piecewise) also skipped
     FULL capture because the skip decision read the *global*
     `compilation_config` instead of the manager's own mode →
     `No cudagraph for ... (cg_mode=FULL, num_tokens=3, num_reqs=3,
     uniform_token_count=1)`.

4. **MRV1 mRoPE/XD-RoPE positions are non-contiguous.** `_get_positions` returns
   `mrope_positions.gpu[:, :num_tokens]`, a view of a `(3, max_tokens + 1)`
   buffer. Inductor asserts the contiguous stride during piecewise capture
   (`expected size 3==3, stride 2048==2049 at dim=0`). This was already
   root-caused in the sibling checkout but left unfixed; MRV2 had the fix
   (`vllm/v1/worker/gpu/mm/rope.py::RopeState.get_positions`).

## Fixes

| File | Change |
| --- | --- |
| `vllm/config/vllm.py` | The piecewise-without-compilation override is now a `warning_once` that names the remedy, and `FULL_AND_PIECEWISE` downgrades to `FULL_DECODE_ONLY` (keeping the FULL graphs it also asked for) instead of dropping every graph. |
| `vllm/models/qwen4_exp/amd/hyperconnection.py` | `_injection_slice()` returns the HC injection slice `.contiguous()`; the non-fused split path does the same. |
| `vllm/v1/worker/gpu/spec_decode/autoregressive/speculator.py` | Draft prefill routes a FULL dispatch to `_prefill(cudagraph_runtime_mode=PIECEWISE)` when `rocm_full_executes_as_piecewise` is true. |
| `vllm/v1/worker/gpu/cudagraph_utils.py` | `CudaGraphManager.capture` only skips FULL capture when the *manager's own* mode has piecewise captures. |
| `vllm/v1/worker/gpu_model_runner.py` | Added packed contiguous mRoPE/XD-RoPE workspaces and `_contiguous_positions()`; `_get_positions` returns a contiguous `(num_dims, num_tokens)` tensor for compiled inputs. |
| `tools/rdna2/serve_v620_{baseline,candidate}.sh` | Added `V620_COMPILE_MODE`, `V620_CUDAGRAPH_MODE`, `V620_CG_SIZES` (defaults unchanged). |

Tests: `tests/compile/test_config.py::test_piecewise_cudagraph_without_compilation`
and `tests/compile/test_cudagraph_replay_inputs.py::test_mrv1_contiguous_positions_per_capture_size`.

## Verified configuration

```bash
bash tools/rdna2/serve_v620_piecewise.sh            # mode 3 + FULL_AND_PIECEWISE
V620_CG_SIZES='[3,6,12]' bash tools/rdna2/serve_v620_piecewise.sh
```

A bare `"cudagraph_mode":"PIECEWISE"` (no FULL graphs) boots the same way and
also serves correct output; its MTP draft-decode steps run eagerly because
the speculator only captures FULL draft graphs.

Boot evidence (per rank):

```
INFO [cudagraph_utils.py:357] ROCm FULL decode executes piecewise CUDA graphs
     (GDN/FA stay eager; inductor FULL replay cannot see new decode inputs).
INFO [model_runner.py:984] Graph capturing finished in 3 secs, took 0.46 GiB
```

Raw server logs: `<wsantos-home>/work/piecewise-c5-full_and_piecewise.log`
(PIECEWISE) and `<wsantos-home>/work/piecewise-d-full_decode_only.log`
(mode-0 FULL baseline).

The speculator captures its own FULL draft-decode graphs. Correctness probes
after boot: `Paris`, `1 + 1 = 2`, `5 + 5 = 10`, `Berlin`; MTP acceptance 80.8 %
(mean accepted length 2.62).

## Measured trade-off (16k / 1024 output, MTP2)

Same box, same harness (`llm-context-bench`, the PR #17 author's protocol), same
launcher parameters; the only difference is `cudagraph_mode`. Both rows are from
*valid* tables (PR #17 cold protocol: `--profile pr17-rocm10-tuned-cold --suite all
--lane performance --sizes 16k --max-retries 0 --input-size-tolerance-percent 11`,
fresh boot, 0.0 % prefix-cache hit rate).

| 16k suite | Prompt tokens | Prefill tokens/s | Generation tokens/s | TTFT s | Valid |
| --- | ---: | ---: | ---: | ---: | --- |
| `FULL_DECODE_ONLY` regular prose | 16,750 | 1,983.8 | 63.07 | 8.44 | Yes |
| `FULL_DECODE_ONLY` coding | 18,063 | 1,985.3 | 70.08 | 9.10 | Yes |
| `FULL_AND_PIECEWISE` regular prose | 16,750 | 1,972.4 | 25.74 | 8.49 | Yes |
| `FULL_AND_PIECEWISE` coding | 18,063 | 1,980.5 | 33.84 | 9.12 | Yes |

Mean inter-token latency doubles: 15.85 -> 38.85 ms (regular) and 14.27 -> 29.55 ms
(coding). Prefill is unchanged within noise: the `[3,6,12]` ladder only covers
decode batches, so prefill runs eagerly in both configurations. On ROCm the
compiled path replays piecewise graphs with eager GDN/attention breaks between
them, while the mode-0 path replays one monolithic FULL graph. For decode-heavy
work the mode-0 `FULL_DECODE_ONLY` configuration remains the faster choice;
compiled PIECEWISE only pays off if the capture ladder covers prefill chunks,
which needs a larger ladder and more VRAM than the fixed 3.75 GiB KV budget
leaves. Graph memory was 0.46 GiB at `[3,6,12]` and 0.37 GiB for mode-0 FULL.

An independent `pr17-bench.py` pass agrees: regular prefill 1,990.3 vs 1,973.8
tok/s and decode 34.7 vs 72.8 tok/s; coding prefill 2,064.1 vs 2,008.1 and decode
37.6 vs 81.3. The plain (non-tolerance) PR #17 harness argv produces
`valid_trials: 0` for both arms — including PR #17's own reference run — so only
the cold protocol gives a comparable table. Artifacts:
`<wsantos-home>/work/llm-context-bench-results/pr17-rocm10-tuned-cold-16k.json`
(mode 0) and `...-cold-full_and_piecewise-16k.json` (piecewise).

## Still open

- **Breakable CUDA graphs (`mode 0` + `VLLM_USE_BREAKABLE_CUDAGRAPH=1` +
  `PIECEWISE`) capture but fault on replay** with
  `HSA_STATUS_ERROR_EXCEPTION` in `indexSelectSmallIndex<Half,long,...>` during
  the eager `qwen_gdn_full_forward` break. Candidate sources: the PLE short-conv
  state indexing (`qwen4_exp/amd/ple_layer.py`) and the GDN spec-decode token
  indexing (`qwen_gdn_linear_attn.py`), neither of which is decorated with
  `@eager_break_during_capture`.
- A ladder that covers prefill chunks (up to `--max-num-batched-tokens 4096`)
  needs a VRAM measurement before it can be adopted.
