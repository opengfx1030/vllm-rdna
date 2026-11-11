# MonoKernel startup fail-stop compatibility check

The `--monokernel` API-config guard now turns on
`VLLM_RAISE_ON_LOGIT_NANS=1` when the option protects K1/K2/MTP and the
variable is unset. An explicit `0` fails config validation; unsupported
platforms and configurations without protected MonoKernel components do not
mutate the variable. This preserves the fail-stop requirement for poisoned
outputs while allowing the documented opt-in launch recipe to start.

## CPU-only validation

All runs below used the frozen 7.14.0 venv on the remote host with zero
visible GPUs. The runner JSON records `cuda_initialized: false`; no serving
source or installed core was changed.

**Pre-change baseline full module:** 29 passed, 3 skipped, 8 failed.
Source hashes: code `47b11839420bbd79aa76d3a970d982db6ffceaa98f298cf8fedce72f74ec3bb9`;
tests `46887db84d5ad27f9327c730babcfd60d0b8d42c4e7e393886df0f20563070bc`.

**Guard-focused selection:** 14 passed, 32 deselected.
Source hashes: code `2cc31d71007e04f29a9105c43f249fa3ccb059d0bf0564c9db1662d3447703ff`;
tests `3fce5df163feb42ad64cf0e728d8c65a823cc1d516bb2a0df1b02f55df914953`.

**Post-change full module:** 35 passed, 3 skipped, 8 failed, using the same
post-change source hashes as the focused run.

All eight full-suite failures are the same test cases in baseline and
post-change runs. Seven test outcomes require GPU-backed imports or setup and
fail because the host exposes no CUDA/HIP devices (three are parameter values
of the same MoE refusal test). The eighth, `test_dense_unquantized_mtp_wrong_shared_shard_falls_back`, asserts the stale word
`geometry`; current code reports the more specific unsupported TP4 shard
dimensions. The changed guard tests passed in the focused run and full module.
A Triton `is_hip_gfx1250` import warning also appears in the MoE import traces,
before the no-device failure.

Evidence files preserve both complete pytest logs and runner JSONs, plus the
baseline and current runner scripts as `.txt` files to protect their exact
executed bytes. The post-change full-module result is
not a clean full-suite pass: the eight known baseline failures remain. Run the
GPU-dependent tests on the intended host before treating the entire module as
validated.

The pre-change suite had 29 passes and the changed suite has 35: the six
additional passing test cases are the new fail-stop compatibility checks.
Failure names and causes match between both full-suite logs.

## Evidence digests

The runner script is SHA256
`b89eb1999cae256758a87fb1ac28b23ed6b3285ba23c62da9de0c09740430e0e` in both
runs. Baseline source/test SHA256 values are
`47b11839420bbd79aa76d3a970d982db6ffceaa98f298cf8fedce72f74ec3bb9` and
`46887db84d5ad27f9327c730babcfd60d0b8d42c4e7e393886df0f20563070bc`.
Post-change source/test SHA256 values are
`2cc31d71007e04f29a9105c43f249fa3ccb059d0bf0564c9db1662d3447703ff` and
`3fce5df163feb42ad64cf0e728d8c65a823cc1d516bb2a0df1b02f55df914953`.
Evidence payload SHA256 digests are recorded in
[`evidence-sha256.txt`](evidence-sha256.txt).
