# V620 candidate integration — 2026-09-13

This remains a development candidate. The initial integration was local only.
After remote testing was authorized, its native extensions were built and tested
in a separate installation on the four V620s. The stable installation is preserved;
it has not been replaced. Historical benchmark results are not candidate results.
The [FP16 performance report](V620-FP16-PERFORMANCE.md) records the current
MTP0–4 comparison, selected settings, loading times and quality limitation.

## Isolation and stable baseline

- Preserved original deployment source: `fa961c2250aa0eb6797b0787b5f04381895c017d` (PR #5).
- Local preservation tag: `stable/v620-fp16-2026-09-12`.
- A checksummed source archive is outside this checkout, under the workspace's
  `stable-releases/v620-fp16-2026-09-12/` directory.
- That local archive contains source only. A separate, verified server snapshot
  now preserves source, built extensions, environment, Python runtime, tools and
  the boot unit at
  `/home/george/v620-stable-releases/2026-09-12-before-refresh`.
- Local testing: `vllm-rdna-testing`, branch `codex/v620-rdna-refresh`.
- Clean PR worktree: `vllm-rdna-pr-clean`, branch `codex/v620-pr-clean`.
- Remote candidate root: `/home/george/v620-vllm-testing`; use `source`,
  `.venv`, build outputs, caches, logs, and any test service exclusively there.
- Protected deployment: `/home/george/v620-vllm/source`, its sibling `.venv`,
  `tools/v620-serve-intel-fp16.sh`, and
  `/home/george/.config/systemd/user/v620-serve-intel.service`.

The runtime snapshot was verified before the remote build.
Keep the original installation at its existing paths; copied virtual environments
can contain absolute paths. Do not change its boot service. Model files can be
read from the existing model directory without editing or redownloading them.

The candidate launcher refuses an environment outside its testing root, refuses
an import from another source checkout, and refuses to launch while another
vLLM process for the same user exists. It never stops a process. It uses test port
8081 by default, with `active` and `qwen3.8-flash-next` aliases. It changes no
client settings. Transient test services are used during authorized validation;
the existing stable boot unit is unchanged.

## Reused changes and preserved behavior

The branch starts directly from
[`opengfx1030:rdna_extras` at c6b5cfb90](https://github.com/opengfx1030/vllm-rdna/tree/c6b5cfb904f2edf99585a8b23c920dbb018ff54e).
It merges subsequent target updates through `f86faadbd` without adding mainline
vLLM history. The replacement contains 20 commits above that target, including
final testing/reporting changes, rather than the original 795-commit history.

| Source | Integrated work |
| --- | --- |
| Current target branch | GDN sequence-local output addressing, state arenas, graph input and collective fixes, gated RMSNorm, current native prefill kernels, and TP4 correctness fallbacks. |
| [PR #6](https://github.com/opengfx1030/vllm-rdna/pull/6), `22bb2e8d0`, `c05af4087` | Attention launch tuning, sequential-layout HIP MoE GEMV, Hybrid W4A16, and MTP/PP plumbing. |
| [PR #8](https://github.com/opengfx1030/vllm-rdna/pull/8), `5765f57b4` | Flash-Next model/PLE/QSA/MTP infrastructure, FP16 hyperconnection fusion, QSA norm/rotary fusion, EP-aware skinny experts and optional dense INT8. |
| [PR #7](https://github.com/opengfx1030/vllm-rdna/pull/7), `361df8a25` | PCIe topology-aware dispatch, combined with newer target all-reduce boot barriers/self-tests. |
| [PR #3](https://github.com/opengfx1030/vllm-rdna/pull/3), `4b9badd0a` | Four-warps/one-stage gfx10 recurrent GDN tuning and shallow fused gating. |
| [Leapdragon](https://github.com/leapdragon/vllm-rdna2-qwen/tree/35b351f5b79f072b9159aba39acd27bbaba25449) | Missing shared-expert fusion caller, rocBLAS build-hash lookup selection and offline TunableOp row probe. |
| Our published branch | CPU PLE materialization/hash/FP8-byte/HIP-runtime fixes, checkpoint expert filtering, startup-plan invalidation, FP16 skinny decode, specialized INT4 expert decode/prefill, QSA live-context scoring bound, rotary/vision support, and separately prefixed draft cache groups. |

The imported code needed integration fixes: missing QSA live-context metadata,
the preselected embedding quantization argument, variable-length Mamba dtype
annotations, and PP drafter typing. Native MoE eligibility now rejects explicit
zero points on **either** projection, since that kernel only implements symmetric
weights. The Intel checkpoint is symmetric INT4/group128; its AutoRound packing
and selected WNA16 runtime path are distinct from the alternative native AWQ
path. It retains the qualified sequential Triton expert implementation.

The target branch's broad FULL-to-PIECEWISE policy is restricted to compiled
configurations that actually have piecewise captures. Compilation mode 0 retains
real full-decode captures. Both graph modes still require runtime qualification.
The corrected native convolution alternatives remain opt-in; the candidate
launcher selects the previously qualified Triton convolution path.

Additional dense INT8, the alternative FP16 dense GEMV, RDNA custom all-reduce,
startup-plan reuse, and TunableOp lookup default off in the candidate launcher.
FP16 hyperconnection/shared-expert fusions remain selectable through the donor
environment switches. Extra dense quantization is not required to use fusion.

## Candidate launch preview

To preview the command from the candidate checkout:

```bash
V620_MM_LIMIT='{"image":4,"video":1}' \
  bash tools/rdna2/serve_v620_candidate.sh --dry-run
```

The image/video counts above are an example for preview, not a qualified capacity
or a proposed permanent limit. Set `V620_MM_LIMIT` explicitly for each memory
test. Context capacity, image resolution and video frame sampling still constrain
what fits in VRAM. Larger multimedia limits require a new profiling/memory check.

The launcher uses FP16, TP4/EP4, MTP1, CPU PLE, a configured 262,144-token context,
4 GiB of KV memory per card, 2,048 scheduled tokens and four request slots, with
decode graph sizes 2/4/8. Tool parsing and thinking-disabled defaults are included.
`VLLM_PLE_QUANT_DIR` may select an existing sidecar; no table is converted by the
launcher. An unset sidecar uses checkpoint PLE in RAM as the comparison baseline.
The explicit Engram config here supports the ROCm worker; embedding-across-DP
sharding is rejected because that implementation has not been ported.

TunableOp remains lookup-only. After offline qualification, set `V620_TUNABLEOP=1`
and `V620_ROCBLAS_LIBRARY` to the actual library loaded by the testing wheel SDK.
The helper requires all four per-device CSVs under
`<test-root>/tunableop/rocblas-<library-sha256-first-12>/`. No donor solution IDs
are assumed valid for the server's different ROCm build. The offline qualifier compares every selected FP16 dense solution with an FP32
reference on all four cards. Full-model measurements and limitations are recorded
in [the TunableOp report](../../tunableop/README.md). Bundled rows are used when
no runtime-generated rows root exists; an explicit `V620_TUNABLEOP_ROOT` overrides
that selection.

## Local validation

- 121 selected CPU tests pass across topology, PLE, expert loading and graph
  policy/input helpers; six GPU cases skip. Five cases are excluded: two model
  download tests, one test requiring the normal configuration fixture, and two
  graph tests requiring accelerator/platform initialization.
- 16 startup-plan persistence/invalidation tests pass.
- Four separately prefixed draft-cache regression cases pass.
- 12 HIP-MoE dispatch predicate cases pass, including down-projection zero points.
- Python syntax, shell checks, repository hooks and static typing are checked
  locally. GPU test collection is separate from GPU execution.

Reproduce the CPU checks without downloading weights:

```bash
.venv/bin/python tools/rdna2/test_startup_plan_cpu.py
.venv/bin/python -m pytest --noconftest -q \
  tests/distributed/test_rdna_p2p.py \
  tests/kernels/quantization/test_rdna2_w4a16_selection.py \
  tests/v1/worker/test_ple_offload_worker.py \
  tests/compile/test_cudagraph_replay_inputs.py \
  tests/model_executor/model_loader/test_ep_weight_filter.py \
  -k 'not torch_compile_matches_eager and not TestSafetensors and not should_copy_and_wrap_eager_piecewise_graphmodules and not mrope_get_positions_contiguous_per_capture_size'
.venv/bin/python -m pytest --noconftest -q \
  tests/v1/core/test_kv_cache_utils.py -k separately_prefixed_draft
.venv/bin/python -m pytest --noconftest -q \
  tests/kernels/quantization/test_rocm_moe_skinny.py -k decode_supported
```

## Still required

1. Stable snapshot and isolated gfx1030 build are complete, using the wheel SDK's
   matching AMD-SMI bindings. Preserve these throughout subsequent experiments.
2. Complete full-model and changing-input graph tests, then short deterministic
   text, tools, thinking, vision/video and MTP checks at concurrency 1/2/4.
3. Establish coherent group-16 INT4 PLE output against the BF16-table control;
   packing/hash/transfer checks alone do not establish model quality.
4. Measure cold/warm API readiness and compare each enabled fusion, QSA profile,
   dense/expert kernel and matched TunableOp lookup against the stable runtime.
   Start with bounded prompts; the published 32k/64k benchmarks are the subsequent
   comparison workloads. Do not begin with a 262k-token prefill.
5. Assess larger graph captures against the full-context VRAM budget. Additional
   dense INT8 needs accuracy measurements before default enablement.
6. Later candidates remain in the reuse audit: mainline PLE/QSA output fusion,
   QSA workspace reuse, portable V2 GDN changes, GPTQ loader exclusions, and
   alternative dense prefill kernels. These are not claimed as integrated here.
7. Keep PR #5 as a draft while the bounded code-answer regression remains
   unresolved. Publishing the clean integration for review does not promote it
   over the preserved stable installation.

## Server regression results

The native suite passes 121 tests. It exposed and now covers shared-memory
write races and padded gate batch strides in the GDN output kernel, and empty
sequence IDs in FLA chunk metadata. QSA native dispatch now checks the tensor
device. The numerical MoE reference models FP16 workspace and output rounding,
with separate finite-input and overflow cases. The production MoE arithmetic
and test tolerances were unchanged by that reference correction.

The first model launch loaded weights but failed because the integration dropped
the stable runtime PLE replication flag. Restoring that flag exposed missing
mixed-state V2 handling. The missing framework portions of upstream
`e126687a9` (PR #53896), plus the copyable-prefix correction in `91752b7a3`
(PR #54634), have been reused from the stable source history. These cover
per-type GDN/PLE state copies, circular-cache slot/prefix handling and speculative
query-row alignment. Validation passes 43 state-copy tests, 5 GPU block-table
tests and 6 scheduling tests using an offline generated config.

The corrected candidate served successfully. Its deterministic reference returned
exactly `blue` with the same 21 prompt token IDs as stable and no reasoning tokens.
Eight answer checks each at concurrency 1/2/4 passed, as did two synthetic vision
checks. Observed readiness was 276 seconds; this is not a controlled cold/warm
comparison because prior failed launches populated parts of the test caches.

Uncached 1024-input/1-output prefill averaged 954.69 tok/s across three measured
requests after warm-up. A random 128-input/64-output test at concurrency 1,
two measured requests after warm-up, produced 40.70 output tok/s including prefill,
20.69 ms TPOT (48.34 decode tok/s excluding the first token), 268.89 ms TTFT and
58.75% MTP acceptance. These results do not establish an improvement over stable.
The 32k/64k comparison and tools/video/broader quality checks remain outstanding.

The run uses the original BF16 PLE table in CPU RAM as the stable control, not a
qualified group-16 INT4 sidecar. Models and their configs are unchanged. The
bounded driver stopped the candidate and restarted the original stable service.

AI assistance was used. Full-model results must be recorded separately from
kernel correctness and historical stable benchmarks.

## Current FP16 performance campaign

Dense INT8 shadows are disabled. The original INT4 expert weights and BF16 CPU
PLE table are unchanged. Current code improvements reuse the donor V620 MoE tile
configuration, qualified FP16 rocBLAS tuning and PCIe push all-reduce. The latter
required correcting its Torch device-context API; the old call silently disabled
that backend. Exact four-rank sums pass repeated changing-input graph replays.

The donor's per-call `wvSplitK` output allocation is restored. A global output
buffer in the target overwrote retained projection results on the next call.
The regression failed on all four cards before the fix and passes afterward;
230 native numerical/lifetime cases pass, including eager and graph FP16/BF16
at batch sizes 1/2/3/4/5. Earlier apparent quality passes using the shared buffer
are not sufficient evidence of correct execution.

The latest target merge preserves the tested GDN initialization and separation
of eager/capture buffers. It does not adopt whole-cache resets or eager writes
into captured storage. The alternative native TP4 W4A16 selector is enabled only
in the breakable graph mode qualified by that upstream change. Our Flash-Next
launch continues to use full-decode graphs and its existing expert/QSA paths.

Post-merge CPU checks pass 133 cases, with ten accelerator cases skipped and
five explicitly excluded model-download/configuration-dependent cases. The
startup-plan suite passes 16 cases. All normal hooks pass on the merge and
subsequent tuning changes. These checks do not replace GPU model evaluation.

At MTP0, the corrected 8k-batch run measured 1,418–1,471 prefill tok/s and
41.98–42.12 decode tok/s across the 16k/32k prose/code tiers. All four performance
trials reached the requested 1,024 output tokens. API readiness took 248.30 s,
including 90.73 s maximum per-worker model loading; this was a cached-filesystem
start. Single/four-request smoke checks pass, but long-context quality is only
2/4: both code cases answer 59 rather than 61 for the fixture's arithmetic.
The untuned corrected FP16 control gives the same failure. This remains an
experimental candidate, with no claim of broad accuracy or BF16 equivalence.

The measured setup uses `Intel/Qwen3.8-Flash-Next-W4A16-AutoRound` revision
`4c67bf686b7f7fd386bae6b07ab59e8ff1d5b897`: AutoRound 0.15.0, symmetric INT4,
group size 128. Original PLE comes from checkpoint shard
`model-00016-of-00017.safetensors`: 128 table shards combine to
`[320001536, 160]` BF16 values, approximately 95.37 GiB in CPU RAM. No quantized
PLE sidecar was used in these measurements. The group16 sidecar is not yet
qualified for coherent model output.

Runtime: Python 3.12.14, Torch 2.13.0+rocm10.0.0, HIP 7.15.26333,
Triton 3.8.0+git4cff872c.rocm10.0.0, Transformers 5.17.0, AMD-SMI
27.0.0+6b0e43f3 from the matching wheel SDK. Installed vLLM version metadata
predates this exported source; use the tested commit and source/native hashes.

The explicit 4 GiB KV allocation already bypasses the profiling that the startup
plan caches; turning that cache on would not further shorten this launch.
