# V620 findings against rdna_extras, September 15

Base: `opengfx1030/vllm-rdna` **rdna_extras**, commit
`3e1a0e1aa0d69de7984c4519b5a110483fa6e982`. This is the RDNA development
branch used as our main branch, not the fork's mainline-tracking `main`.

## Fixes included

- Materialize CPU-owned PLE subtrees during meta model discovery. Otherwise
  checkpoint copies into meta parameters silently retain no table data.
- Preserve the original BF16 CPU PLE table when model activations use FP16.
- Construct GDN gated normalization on `A_log.device`, respecting meta discovery.
- Pass the required `w2_zp` argument at all three skinny MoE call sites.
  The omission caused startup warm-up to fail even with skinny MoE disabled.
- Only redirect ROCm FULL replay to piecewise graphs when compilation and
  piecewise captures are enabled. Direct FULL capture with compilation mode
  NONE is required by this V2/MTP2 launch; unconditional redirection caused
  warm-up to request an uncaptured graph.

These are the corrections applied to the isolated upstream trial. No native
kernel changes are included. AI assistance was used; these changes still need
human review before an upstream contribution.

## Trial configuration and evidence

Four V620s; Intel/Qwen3.8-Flash-Next-W4A16-AutoRound revision
`4c67bf686b7f7fd386bae6b07ab59e8ff1d5b897`, existing local checkpoint;
original BF16 CPU PLE, INT4/group128 experts, FP16 dense weights. Python3.12,
Torch2.13.0+rocm10.0.0, matching wheel SDK AMD-SMI bindings.

TP4/EP4, V2 runner, MTP2, batch4096, block1024, max context262144,
4GiB KV per GPU, FULL_DECODE_ONLY captures[3,6,12], compilation mode0.
Qualified FP16 TunableOp rows enabled for rocBLAS c27e2252cc7a.
RDNA all-reduce explicitly enabled,64KB cap, automatic blocks, pacing0;
this differs from the target's default collective policy.

Launch adaptations outside the model fixes:

- `VLLM_PLE_CPU_OFFLOAD=1`: target has no `--engram-config` CLI despite the
  copied V620 launcher's use of that option.
- `--kernel-config '{"moe_backend":"triton"}'`: startup confirms
  TritonWNA16Experts for this Intel packing.
- `--no-enable-prefix-caching`: upstream documents recurrent-state corruption.
  CPU KV offloading was omitted for this trial.

Fresh native build registered67 ROCm schemas. CPU PLE reproduction retains
checkpoint rows after the fix; real GDN constructor tests passed2/2; graph
selection tests passed8/8; all3 MoE calls bind to the current signature.
The third launch completed MTP warm-up and returned healthHTTP200 after
379.49seconds. Earlier failures were retained, not counted as successful runs.
**No generation requests, output-quality evaluations or performance benchmarks
were run. Readiness is not proof of coherent output or a speed improvement.**
The normal service and its build were preserved; boot default was not changed.

## Remaining findings, not fixed by this branch

- Prefix-cache corruption: see `flash-next-prefix-cache-corruption.md`.
  The published reproducer uses V1; V2 is not independently cleared.
- RDNA2 WNA16 oracle integration: the new backend is selected automatically,
  but the factory allowed-experts tuple and format conversion lack matching
  support. Its layout must be qualified for Intel AutoRound before use.
- GDN prefill output's non-varlen gate batch stride does not include PR5's
  `g.stride(0)` handling. Port and test padded/non-contiguous batches separately.
- `wvSplitK` still returns shared static output storage, unlike PR5's retained
  output fix. Do not enable that conditional decode path without restoring
  and testing the lifetime correction. The target's gfx1030 default uses GEMV.
- Repair the copied V620 launcher and documentation to use current CPU-offload
  configuration. A generic launch should not silently impose this trial's
  Intel-specific backend and cache workaround.

The maintainer explicitly imported selected changes, not all PR5 code:
[PR5 closing comment](https://github.com/opengfx1030/vllm-rdna/pull/5#issuecomment-5660839758).

## Branch checks

Local graph-selection regression:8passed. Local CPU PLE checkpoint-retention
regression:1passed. Commit-time checking additionally found the same missing
`w2_zp` argument in `moe_skinny_decode_supported`; that fourth call is corrected
here, but was not part of the already-running server's startup trial.

The unmodified base reproduces failures in ruff-check, ruff-format, typos,
mypy-3.10 and check-torch-cuda-call on the touched upstream files. Those hooks
were skipped for this focused commit after recording the baseline failures;
other applicable hooks passed. Existing whole-file formatting/type issues
were not folded into these runtime fixes.
