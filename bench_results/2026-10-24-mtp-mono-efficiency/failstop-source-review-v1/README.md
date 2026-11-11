# Mono MTP fail-stop source and validation review

This bundle pins the 11-file candidate source overlay reviewed for the
fail-stop change and records how the formatted working tree was compared with
the exact candidate used for the disposable tests. It contains source text,
hashes, and validation logs only; it contains no binaries or code objects.

## Source identity and equivalence

- The candidate overlay source hashes are preserved in
  `validation/proposal-source-sha256.json`; all 11 exact candidate files are
  under `raw-candidate/` as `.txt` snapshots.
- The tested extension/core hash is
  `6facdbd60118fc14a5cf1ceb5c2494d38ed1be09f55e4636bb0f0825678e2eed`.
- `verify_postformat_source.py` compares candidate snapshots with the current
  working tree without importing Torch or running a compiler. It confirms
  Python AST identity and C++ token identity, including C++ adjacent string
  literal concatenation and comments. The latest result is
  `postformat-equivalence-current.json`.
- `validation/postformat-equivalence-original.json` is the earlier recorded
  mapping; it predates comment-only updates and is retained for provenance.
  Use the generated `postformat-equivalence-current.json` for current local
  hashes.
- `validation/applied-validation.md` summarizes the applied patch and gates;
  `validation/precommit-current.log` records hook results. The only failing
  hook is mypy on the existing `hidden_states.shape` nullable error. The same
  diagnostic reproduced against baseline commit `a280b4f507`; no suppression
  or unrelated runtime fix was added. See `validation/mypy-baseline.log`.

## Tests

The direct kernel suite passed 289 tests, with 48 skips and 14 warnings; the
eager fault-containment engine check and independent privileged journal audit
are recorded separately under
`../failstop-engine-eager-review-v1/`. The newly added CPU contract test
`tests/models/qwen4_exp/test_mono_mtp_failstop.py` was tested as part of a
7-pass/14-warning CPU helper run. Its tested source SHA-256 is
`7af7fbb38ee7c6da7e38bc0d13192e15f3ee997cdeb65bca1a334d29152978fe`; the
source itself remains in the normal test path rather than duplicated here.

These checks do not yet constitute graph-mode engine qualification or a
serving performance claim. The graph test was being run separately when this
bundle was prepared.

## Re-run the source equivalence check

From the repository root, use the pinned validation interpreter:

```sh
/private/tmp/mono-local-validation-venv/bin/python \
  bench_results/2026-10-24-mtp-mono-efficiency/failstop-source-review-v1/verify_postformat_source.py.txt
```

The command is CPU-only and rewrites only the generated
`postformat-equivalence-current.json` inside this evidence directory.
