# Mono MTP fail-stop graph-mode cleanup qualification (v3)

This is a compact, response-free record of one injected-fault qualification. The fault was deliberately raised on TP rank 1 while Mono MTP K2 graph execution was enabled. The driver recorded ARM-PASS: the injected request and two subsequent requests returned HTTP 500, the EngineCore fail-stop error propagated, all four worker PIDs were observed exited before the scoped driver cleanup completed, and the pinned privileged kernel-journal checker found 79 newly appended records with no GPU-fault/reset/trap match. The before/after boot IDs and source/core hash manifests are identical.

This result validates fault propagation and worker/process cleanup for this one graph-mode run. It does **not** show the initial kernel barrier failure was prevented, does not establish recovery without restarting the engine, does not qualify normal generation correctness or performance, and does not establish behavior across other schedules or hardware. HTTP 500s are expected in this injected-fault scenario.

The exact large journal snapshots and full server log are omitted because they contain unrelated host records and request/debug payloads. `kernel-journal-audit-summary.json` records their counts and the driver's pinned pass result; `qualification-events.txt` retains only safe event lines. Raw sources are represented by `sha256.before` and `sha256.after`; they are byte-identical. `worker-provenance.json` maps TP ranks 0–3 to the expected physical KFD GPU IDs and the same loaded core SHA.

No binary, cache, prompt/response dump, or raw kernel journal is included. The test driver source and full raw logs remain at the corresponding local run path outside this compact bundle.
