# Explore: RDNA2 W4A16 M-dispatch under torch.compile

**Status**: explore. It is not known yet whether the serving path is
affected, or what fixing it would change. Nothing changes by default:
`VLLM_RDNA2_W4A16_RUNTIME_DISPATCH=1` opts in to the candidate fix for A/B
runs on the V620.

## The question

`RDNA2W4A16LinearKernel.apply_weights` calls `_rdna2_w4a16_select_kernel`
with `M = x.size(0)` in Python and runs one of three ops:

| M | AWQ (`uint4`) | GPTQ (`uint4b8`) |
| --- | --- | --- |
| ≤ 32 | `rdna2_decode` if K ≥ 4096, else `prefill` | same |
| 33–256 | `prefill` | `exllama` if N ≥ 3072, else `rdna2_decode` |
| > 256 | `prefill` | `exllama` |

vLLM compile traces the model once, in the profile run at
`max_num_batched_tokens`, and then runs the compiled code with Dynamo's
guards dropped. `TorchCompileWithNoGuardsWrapper` skips every guard unless
`dynamic_shapes_config.evaluate_guards` is set (default `False`); with
`VLLM_USE_BYTECODE_HOOK` it calls the compiled bytecode directly. A Python
branch on `x.size(0)` is then evaluated once, at trace time. The graph
would hold one op per GEMM, and every batch size, including captured FULL
decode graphs, would run it.

JartX hit this pattern on gfx1100 with guards kept: Dynamo guarded every
layer and decode got 7× slower
([README_RDNA3.md, lesson 2](https://github.com/JartX/vllm/blob/dafcde3f8bb9da96e5ca24adcf8f54c9ced413d7/csrc/libtorch_stable/quantization/gptq/README_RDNA3.md)).
With guards dropped the failure would be silent: no recompiles, just the
trace-time kernel everywhere.

## Evidence so far (no GPU)

1. **Mechanism, on the real dispatch code.**
   `test_rdna2_w4a16_dispatch_under_vllm_compile` in
   `tests/kernels/quantization/test_rdna2_w4a16_selection.py` compiles
   `_rdna2_w4a16_gemm` with vLLM's guard policy and tagged CPU stand-ins
   for the three ops. Traced at M=512, an M=1 batch that the selector sends
   to `rdna2_decode` runs `prefill` (AWQ) or `exllama` (GPTQ), with one
   compile and no recompile. Through the custom op it runs `rdna2_decode`.
2. **The repo already avoids this elsewhere.** `rocm_unquantized_gemm_impl`
   (`vllm/model_executor/layers/utils.py`) branches on the token count,
   including the gfx10x `gemv_f16_rdna2` path for M ≤ 8, inside a
   `direct_register_custom_op` op that Dynamo cannot see into.
3. **A hint in the decode profile.** In
   `docs/profiling/2026-09-10-decode-kernel-profile.md` (FULL_AND_PIECEWISE,
   27B AWQ, TP=2), the prefill op's `gemm_dynamic_kernel` ran 19,872 times
   at c=1 (M=1), against 4,608 for the decode op's
   `gemm_q4_kernel_rdna2<__half, 1>`. That fits a frozen `prefill` choice
   if most of the model's GEMMs have K ≥ 4096. It does not prove it:
   something still launched the decode op at M=1.

## Hypotheses and kill criteria

| # | Hypothesis | Check | Holds if | If not |
| --- | --- | --- | --- | --- |
| H1 | The compiled graph holds one W4A16 op per GEMM, fixed by the trace-time M | `probe run` with the serve compile config (B1) | only the op chosen at `trace_m` appears, e.g. only `gptq_gemm_rdna2_prefill` for AWQ | stop: dispatch already follows M on the box |
| H2 | Decode therefore runs other kernels than the selector intends | rocprofv3, compiled arm against `--enforce-eager`, same workload (B2) | decode W4A16 kernel mix differs between the arms | stop |
| H3 | Dispatching per call is faster where the choice differs | serve matrix, `VLLM_RDNA2_W4A16_RUNTIME_DISPATCH=0/1` (B3) | decode tok/s ≥ +3 % at c=1 or c=8, prefill within ±2 %, greedy output identical, GSM8K within noise | keep today's path and note that the frozen choice is acceptable |

## Candidate fix in this PR (opt-in)

- `_rdna2_w4a16_gemm` holds the existing selection and op calls, unchanged.
  The default path still traces it inline, so it compiles as before.
- `torch.ops.vllm.rdna2_w4a16_gemm` registers that function with
  `direct_register_custom_op`; its fake returns `[M, N]`. Dynamo records a
  single opaque op, and the choice happens per call when eager and once per
  size during graph capture.
- `VLLM_RDNA2_W4A16_RUNTIME_DISPATCH=1` routes `apply_weights` through the
  op; `0`, the default, keeps today's path.
- JartX moved dispatch into the C++ op entry. The Python custom op is this
  repo's existing pattern and needs no rebuild; move to C++ later if the
  per-call Python cost shows up in eager runs.

## Risks to check on the box

- **Shared output buffers.** Each W4A16 op returns a view of one
  process-wide `Rdna2PersistBuf` (`csrc/rocm/rdna2_graph_keepalive.cuh`),
  with one capture slot and one eager slot per op. If H1 holds, captured
  decode graphs use only the trace-time op's buffer today. Dispatching per
  call brings the decode op's buffer into the graphs, a combination that
  has not been exercised. A greedy check under FULL_AND_PIECEWISE and a 16k
  prompt must pass before any timing counts; 16k prefill is where the
  persist-buffer bugs showed.
- **Thresholds.** The selector's M/K cut-offs come from eager microbenches
  (`q_gemm_rdna2_prefill.cu` envelope: M ≤ 32 and K < 4096). If H1 holds,
  compiled serving has been running outside that envelope, and the cut-offs
  have never been checked under graphs. H3 is that check.
- **Compile cache.** The cache key does not include the env var, so arms
  sharing a `VLLM_CACHE_ROOT` would share compiled graphs. `probe run` uses
  a fresh cache per arm; do the same for serve runs.

## Run plan (V620)

Record the git SHA, torch and ROCm versions, board, power cap, compile
config and `max_num_batched_tokens` for every run. Results go in
`RESULTS-<date>.md` next to this file.

### B0. CPU checks on the box

```bash
python -m pytest tests/kernels/quantization/test_rdna2_w4a16_selection.py \
    benchmarks/kernels/w4a16_compile_dispatch -q
```

### B1. H1: what the compiled graph calls

```bash
M=benchmarks.kernels.w4a16_compile_dispatch.probe
MODEL=/models/Qwen3.8-27B-AWQ-INT4
python -m $M run --model $MODEL --tp 2 --json python.json
VLLM_RDNA2_W4A16_RUNTIME_DISPATCH=1 python -m $M run --model $MODEL --tp 2 --json op.json
python -m $M run --model $MODEL --tp 2 --enforce-eager --json eager.json
python -m $M compare python.json op.json eager.json
```

`run` defaults to the compile config of `tools/rdna/serve_gfx1030_full.sh`
(`--compilation-config` to change it). It counts W4A16 op calls in every
`computation_graph.py` of that arm's compile cache and prints the
selector's expected choice per M for the model's real layer shapes. It also
records greedy outputs and median-of-3 throughput for c=1 and c=8 decode
(256 tokens) and a 2k-token prefill.

| Arm | Ops in compiled graph | Expected at `trace_m` | Expected at M=1 | Greedy == python arm | c=1 tok/s | c=8 tok/s | 2k prefill tok/s |
| --- | --- | --- | --- | --- | ---: | ---: | ---: |
| python (today) | | | | — | | | |
| op (`RUNTIME_DISPATCH=1`) | | | | | | | |
| eager | — | | | | | | |

### B2. H2: which kernels decode launches

Use the rocprofv3 recipe of the decode profile (venv-bundled rocprofv3,
`--kernel-trace`), once per arm, around the same workload:

```bash
rocprofv3 --kernel-trace true --output-format csv --output-file python -- \
    python -m $M run --model $MODEL --tp 2 --repeat 1
python -m $M kernels python_kernel_trace.csv eager_kernel_trace.csv op_kernel_trace.csv
```

| Arm | `rdna2_decode` calls / ms | `prefill` calls / ms | `exllama` calls / ms |
| --- | --- | --- | --- |
| python | | | |
| eager | | | |
| op | | | |

Also repeat B1–B2 with a GPTQ (`uint4b8`) dense model if one is served:
there the frozen choice would be `exllama` at every M.

### B3. H3: serve matrix

Run `docs/rdna2/bench_27b_awq_matrix.md` (1k/512 and 16k/1k, c=1/4/8) with
`VLLM_RDNA2_W4A16_RUNTIME_DISPATCH=0` and `=1`, a fresh `VLLM_CACHE_ROOT`
each. Then run GSM8K through `tests/evals/gsm8k` on both.

| Cell | tok/s, 0 | tok/s, 1 | Δ | TTFT, 0 | TTFT, 1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1k/512 c=1 | | | | | |
| 1k/512 c=8 | | | | | |
| 16k/1k c=1 | | | | | |
| 16k/1k c=4 | | | | | |

| Check | 0 | 1 |
| --- | --- | --- |
| greedy prompts identical | | |
| 16k prompt coherent | | |
| GSM8K (5-shot, 500) | | |

## If it graduates

Flip the default (or remove the flag and the inline path), move the tests'
expectations accordingly, and consider the same check for other Python
dispatchers that branch on the token count inside traced code.
