# Direct native kernel suite

- Result: **289 passed, 48 skipped, 14 warnings**, 101.02 seconds.
- Core schema reported PASS and 141 registered operators, including
  `all_reduce`.
- The suite used the candidate disposable overlay and completed on the same
  host boot as the build and collective microcheck.
- Suite coverage includes K1/K2 reset semantics, new sticky-error behavior,
  K2→K1→K2 ordering, GDN, and QSA cases.
- This is a direct kernel/operator suite. It does not exercise a complete
  inference engine or establish serving-level correctness/performance.

See `pytest.log`, `inputs.before`/`inputs.after`, and overlay/boot snapshots in
this directory for the preserved run evidence.
