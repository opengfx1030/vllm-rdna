# V620 RDNA all-reduce: matching native build

## Status

Server qualification began on September 27 in an isolated Git worktree after
the user granted an exclusive test window. The shared runtime is unchanged.
The failure observed on September 27 was a missing
`torch.ops._rocm_C.rdna_ar_timeout_info`, not a proven peer-to-peer hardware
fault. Its definition and Torch registration already exist in this checkout.
The process loaded `_rocm_C` from a different runtime source directory.

The change checks the six required native operators before peer-buffer
allocation, with all ranks agreeing on fallback. General vLLM serving still
falls back to RCCL for an incomplete native build. The V620 baseline launcher,
which explicitly requests RDNA AR, instead fails preflight before loading the
model if an operator is missing or Python/native imports come from another
checkout. A valid binary must reside in the selected checkout's `vllm/` directory;
a symlink resolving to a foreign runtime does not satisfy this check.

The preflight verifies operator availability and import location, not GPU
correctness or exact binary reproducibility. It does not bypass timeout
protection, delete wedge markers, change the 64 KiB cutoff, or enable native
collectives after a failed self-test.

## Isolated native build

Stop the current service only within an approved server build/test window.
Use a separate Git checkout and the matching ROCm/PyTorch toolchain. Do not
overwrite the shared runtime's loaded `.so`, copy a binary from another revision,
or change vLLM source on the server. Commit/push locally, then pull the exact
branch on the server before building.

With `source_root` pointing to that isolated Git checkout, `runtime_python` to
the matching `.venv/bin/python`, and `rocm_root` to the matching compiler/SDK:

```bash
build_root=$(mktemp -d /tmp/v620-rdna-ar-build.XXXXXX)
export PYTORCH_ROCM_ARCH=gfx1030
export ROCM_PATH="$rocm_root"
cmake -S "$source_root" -B "$build_root" -G Ninja \
  -DVLLM_TARGET_DEVICE=rocm \
  -DVLLM_PYTHON_EXECUTABLE="$runtime_python" \
  -DCMAKE_HIP_ARCHITECTURES=gfx1030 \
  -DROCM_PATH="$rocm_root" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$source_root"
cmake --build "$build_root" --target _rocm_C --parallel 4
cmake --install "$build_root" --component _rocm_C
PYTHONPATH="$source_root" "$runtime_python" \
  "$source_root/tools/rdna2/check_rdna_ar_native.py" \
  --source-root "$source_root"
```

This incremental CMake workflow, scoped to `_rocm_C`, built successfully with
the server's ROCm 10 SDK and existing PyTorch runtime. The server additionally
required its SDK library search paths and `TRITON_KERNELS_SRC_DIR` pointing to
the existing runtime's dependency directory. Do not resolve build failures by
modifying the active service's source files.

## GPU findings (September 27)

### Test provenance

The GPU and model measurements below were collected on
`codex/rdna-decode-performance-investigation`, which also contained the RAM KV
offload changes proposed in
[PR #24](https://github.com/opengfx1030/vllm-rdna/pull/24). The commit identifiers
below refer to that original tested history. Model-level RAM reload observations
therefore describe the combined experimental stack, not this all-reduce change
alone.

The review branch, `codex/rdna-ar-correctness`, cherry-picks only the all-reduce
fixes, qualification tools, and this report onto `rdna_extras` at `3d67bdf49`.
It does not include PR #24 or change a systemd service. The focused 16-test CPU
suite was rerun on this clean branch; GPU/model qualification has not been
repeated on the newer base. Re-run the hardware gates before deployment.

### Output ownership and collective qualification

The rebuilt binary exposed a second, independent bug: the native all-reduce
returned views of one process-global persistent output. Two consecutive calls
overwrote the first result on all four ranks. The committed qualification
harness reproduced this twice before the fix. Commit `53b0b10de` restores
per-call output ownership with `at::empty_like`.

The corrected build passed all nine FP16/FP32 payload cases through 64 KiB:
retained eager outputs, two consecutive captured collectives, 16 graph replays
with changing inputs, interleaved eager work, and rejection above the cutoff.
All four native startup self-tests passed. No peer timeout was reported.

Run the committed harness using the matching runtime and library paths:

```bash
PYTHONPATH="$source_root" "$runtime_python" -m torch.distributed.run \
  --standalone --nproc-per-node=4 \
  "$source_root/benchmarks/kernels/benchmark_rdna_ar.py" --repeats 100
```

The clean latency run at `6ed0f5e89` used two out-of-place collectives per graph,
five samples of 100 replays, and the maximum per-rank median. RCCL uses PyTorch's
NCCL process group; these are microbenchmarks, not end-to-end vLLM decode rates.

| FP16 bytes | RDNA µs/collective | RCCL µs/collective |
| ---: | ---: | ---: |
| 5,120 | 30.8 | 82.1 |
| 15,360 | 43.4 | 82.8 |
| 30,720 | 51.2 | 76.5 |
| 61,440 | 110.2 | 63.2 |
| 65,536 | 103.7 | 62.7 |

These results motivate testing a 32 KiB cutoff, leaving larger reductions on
RCCL. The launcher now respects explicit `VLLM_RDNA_AR=0` and
`VLLM_RDNA_AR_MAX_KB=32` settings for controlled model comparisons; defaults
remain unchanged. An initial timing run completed measurements but stalled
during RCCL teardown because the harness retained the final captured graph.
Resetting captured graphs before destroying the group fixed that harness issue;
the repeated run exited successfully.

A final microbenchmark at `c88b2bc8b` also compared the actual vLLM
`PyNcclCommunicator` path, not only PyTorch's process group. All correctness
checks and cleanup passed. FP16 RDNA versus direct-vLLM RCCL latencies were
30.4/79.1 µs at 5,120 bytes, 44.1/79.4 µs at 15,360 bytes, 54.2/74.4 µs at
30,720 bytes, and 117.0/61.0 µs at 61,440 bytes. The small-message benefit is
real in isolation, but does not establish a model-throughput improvement.

Server evidence is under
`/home/george/v620-experiments/rdna-ar-results-20260927/`:
`correctness-before.log`, `correctness-before-repeat.log`,
`correctness-output-fix.log`, and `latency-qualified.log`.
The direct-vLLM comparison is in `latency-vllm-rccl.log`.
Model-level qualification is still required before changing the default service.

### Intel model qualification

The isolated Intel AutoRound service used the rebuilt extension, a 32 KiB RDNA
cutoff, MTP2, FULL_DECODE_ONLY graphs, and 64 GiB RAM KV offload. Both TP and EP
groups selected `RDNA_ONESHOT` before `PYNCCL`. Three simultaneous short chats
passed exact tool-name/argument checks and resumed correctly after staggered
synthetic tool returns. RAM KV loads occurred during those continuations.

`llm-context-bench` coding performance results (one measured repetition, 1,024
output tokens, 12% input-token tolerance):

| Input tier | Actual prompt tokens | RDNA decode tokens/s | RCCL decode tokens/s | RDNA / RCCL TTFT seconds |
| --- | ---: | ---: | ---: | ---: |
| 8K, one chat | 8,964 | 68.37 | 64.23 | 6.43 / 6.43 |
| 64K, one chat | 71,991 | 65.94 | 65.93 | 49.97 / 49.63 |
| 128K, one chat | 143,875 | 64.09 | 66.34 | 103.88 / 103.67 |

All six single-session trials passed benchmark validation. The RCCL baseline
used the same native build, model, graph settings, warmed compilation cache,
and RAM offload capacity, with only RDNA dispatch disabled. These single-trial
measurements use unique request tags; generated outputs and MTP acceptance
can vary. They do not demonstrate a consistent decode gain: observed changes
were +6.5%, effectively zero, and -3.4% respectively.

The three-chat 8K run completed all requests, but one response failed the
benchmark's dominant-token gate: 217 hyphens in generated comment/copyright
headers exceeded the 20% threshold. The aggregate group remains invalid;
do not present it as a validated throughput result or weaken the validator.
The matched RCCL three-chat run passed all three outputs: 22.16 decode tokens/s
per stream (median), 56.24 aggregate tokens/s over the shared generation window.
Do not compare that valid group with the rejected RDNA group as a speedup claim.
These performance checks do not establish code-quality correctness.

Cold-cache startup spent several minutes compiling Triton GDN prefill variants,
confirmed by a worker stack sample in `make_amdgcn`. During the 128K request,
a long GPU wait was initially suspected to be a stall; the request completed
normally and passed validation before the service was stopped for comparison.
No RDNA wedge marker was produced. Long prefill pauses alone are not evidence
of a deadlock.

Full-VRAM multi-session offload stress remains a separate qualification step.
The experiment does not establish a consistent production performance gain;
do not promote RDNA to the default service on this evidence alone.

## Qualification gates before activation

1. Preflight reports the intended checkout's Python and native extension,
   including `rdna_ar_timeout_info`. Record Git revision, binary hash and
   compiler/runtime versions.
2. All four ranks pass the native startup self-test; logs select RDNA_ONESHOT
   rather than only PYNCCL. Never make a missing timeout operator return zero.
3. Test FP16/FP32 exact sums with changing inputs, small/decode-shaped messages
   through 64 KiB, and fallback above that limit.
4. Verify FULL graph capture and repeated replay with changing inputs,
   consecutive collectives, retained outputs, and interleaved eager work.
   Confirm timeout checks remain active. CPU mocks do not establish these.
5. Compare otherwise-identical RCCL and RDNA AR runs at concurrency 1/2/3/4,
   including MTP2, tool returns, mixed prefills, and long-context RAM reload.
   Report per-session latency and aggregate throughput, not a single token/s
   sample. Only then consider activation.

Local regression command (no HIP required):

```bash
.venv/bin/python -m pytest --noconftest tests/distributed/test_rdna_ar.py -q
```

The focused suite isolates the communicator with lightweight vLLM stubs while
using real PyTorch namespaces. It covers abort handling, markers, per-rank
missing operators, and strict preflight import provenance. Heavy repository
conftest fixtures are intentionally excluded. The suite passes 16 tests;
model-level validation remains separate from these CPU and collective checks.
