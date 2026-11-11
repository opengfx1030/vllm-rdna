# First MoE offload optimization pass — 2026-10-29

Runtime change: `37360d8144`; cold-prefix benchmark gate: `be26661733`.
The tested source/native binary hashes and server CPU are in
`runtime-manifest.txt`. Only HIP GPUs 6–7 were used for this pass.

## Release evidence

- `release-gates.json`: all final gates passed.
- `kernel-tests.txt`: 147 focused host/GPU cases passed, including real CT
  overflow/stream parity for tensor and native copies.
- `graphs-cold/`: TP2/EP2 Flash-Next AWQ text serving, MTP2, segmented
  FULL_AND_PIECEWISE graphs, 96 expert slots/layer, 3.75 GiB GPU KV/rank,
  and 16 GiB aggregate CPU KV. Greedy, PPL (4.82 native / 4.83 tensor), prefix
  reuse and 48 normal-EOS concurrency checks per backend passed. The ABBA
  matrix completed 72 exact-length requests and 55,296 output tokens.
- `eager/`: native-copy eager fallback, PPL 4.84, greedy, prefix reuse,
  48 normal-EOS concurrency checks, 1k/512 and 16k/1k c1 exact-length workloads,
  and post-load greedy passed. Graph mode NONE is confirmed in the serve log.
- Both runs passed post-load correctness and reported zero numeric/new GPU
  fault markers. Concurrency reports retain near-tie and soft variations;
  there were zero hard failures.

## Performance scope

`graph-performance-summary.txt` reports all four cells' prefill, decode,
aggregate output, TTFT and ITL ranges. JSON additionally retains MTP acceptance,
expert demands/misses and copied bytes per arm. Both GPU and CPU prefixes reset
before every arm; hit counters are verified zero. Expert histories and MTP
trajectories still differ, so these are observations, not a causal or universal
copy-backend speedup claim. Tensor copies remain the default.

`map-remote.json` measures the actual CPU `PackedExpertCache.ensure/release`
resident-hit path on the build server. At ten active experts its median fell
from 83.42 to 37.73 microseconds (30 ABBA samples per arm). This excludes GPU
fences and DMA time. `map-local.json` is the local CPU sanity run and is not
used as the server result.

## Reproduction

Use the saved graph/eager serve scripts and load hooks with the isolated venv
and cached checkpoint described in the project documentation. Run
`GPU=7 bash tools/rdna/mono/test_offload.sh` for the kernel gate and
`.venv/bin/python -m pytest --noconftest tests/models/qwen4_exp/test_offload_long_probe.py`
for the client gates (28 passed).

For the CPU comparison, extract the reference cache source with
`git show 8492a87f84:vllm/model_executor/layers/fused_moe/rdna_expert_cache.py`
into a file, then invoke `map-benchmark.py.txt` with `--repo`, `--reference`
and `--out` through the venv Python. GPU visibility must remain restricted to
6–7 even for a CPU benchmark because the host's sitecustomize can initialize HIP.
`summarize.py.txt` regenerates the graph summary from this directory.

The earlier NaN investigation, isolated GDN fix and failed pre-fix receipt
remain in the 2026-10-28 receipts. The failure was not causally attributed;
subsequent diagnostic replay and this final qualification passed without
recurrence. Hosted workflow checks were skipped; local hooks and dedicated
hardware tests supplied the validation. Older HIP runtimes and vision inputs
are not qualified by this text-serving pass.
