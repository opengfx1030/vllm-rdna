# Applied proposal validation

The 11-file K1+K2 proposal is applied locally. The disposable runtime tree uses
the pinned raw proposal; local formatting has identical Python AST and C++
lexical tokens (including adjacent string-literal concatenation). See
`postformat-equivalence.json`. The new helper also adds the required SPDX
copyright comment. No raw proposal source was changed during remote tests.

All current source hooks pass except the existing nullable `hidden_states.shape`
mypy error in V2 warmup. The same error reproduces on commit a280b4f507 before
this proposal (baseline line 934; current line 939). See `mypy-baseline.log` and
`precommit-current.log`. No suppression or unrelated runtime change was added.

TP4 health transport passes with installed old3f688 core: actual coarse HSA
MTP arena, one-rank errors, immutable inputs, separate output lifetimes and
changed-input graph replay. That validates transport, not engine shutdown.
The new O2 core6fac is isolated and uninstalled; direct and engine gates are
recorded separately as they complete. No serving performance gain is claimed.
