# MonoKernel fail-stop eager engine check

This bundle records one eager-mode, TP4 engine fault-injection check and a
small CPU-only helper test. It is a bounded correctness check, not complete
engine qualification or serving-performance evidence.

## Eager engine check

`engine-eager/rank1-qualification.json` is the original qualification result.
It records an injected rank-1 request returning HTTP 500, followed by two
later HTTP 500 replies. All four worker processes exited, source/core/boot
hashes were unchanged, and cleanup ownership checks passed. The test's
acceptance criterion was clean fault containment and teardown; it does not
show that later requests recover and serve successfully.

The original `kernel.before.log` and `kernel.after.log` were permission-limited
and had no readable kernel entries, so they are **not** used as evidence that
the host had no GPU faults. Instead, `engine-eager/kernel.privileged-window-audit.json`
records a separate privileged journal read over the full lifecycle window:
133 readable kernel records from 09:41:43 to 09:45:37 CEST and zero matching
broad GPU/PCIe fault patterns. The raw journal is excluded; its SHA-256 is
retained in the audit JSON.

The run used host boot
`1e8b5da1-759f-4dbb-ae87-e58c2faaff48`; the before/after files preserve that
and the source/core hashes. Model/config and four-rank worker provenance are
included without server logs or generated response text.

## CPU helper tests

`cpu-helpers/pytest.log` records **7 passed, 14 warnings in 6.61 seconds**.
This suite covers helper logic only and does not initialize a GPU.

## Qualification boundary

The graph-replay engine qualification is still running separately. No graph
result is represented here. This bundle makes no performance claim and does
not establish full engine qualification.

## Retained artifacts

Only the original small qualification/provenance JSON, source hash manifests,
privileged audit summary, and CPU helper test records are retained. Raw
generated-response/server logs, raw host journal, binaries, and code images
are excluded. `evidence-manifest.json` hashes all files in this curated bundle.
