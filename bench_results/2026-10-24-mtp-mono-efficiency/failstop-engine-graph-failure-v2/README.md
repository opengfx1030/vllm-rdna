# Graph-mode injected-fault run: cleanup gate failed

This is a compact record of one TP4 graph-mode fault-injection attempt on boot `1e8b5da1-759f-4dbb-ae87-e58c2faaff48`. The intended rank-1 MTP K2 fault surfaced as an engine-dead error and HTTP 500 responses, but the qualification script found all four original TP worker PIDs alive at its worker-exit check. The graph-mode worker-cleanup gate therefore **FAILED**.

The retained excerpt shows the fatal error, the executor beginning its bounded worker wait, and the API's immediate shutdown/SIGTERM sequence in the same second. This is consistent with a signal/cleanup race, but the logs do not establish which instruction received SIGTERM or prove the cause. Later process exit state was not captured by this qualification artifact.

This run does not qualify graph-mode recovery, subsequent requests, or cleanup. The engine failed closed to callers; do not treat the HTTP 500 as successful recovery. The cleanup proposal is recorded separately in `../failstop-engine-cleanup-cpu-review-v1/` and has only CPU signal-ordering tests so far.

## Scope and retained evidence

- `failure.txt`: exact qualification traceback plus selected server-log lines and source-file/line references.
- `run-metadata.json`: boot, model snapshot identifier, TP worker/core identity, fault marker, and captured failure status.
- `source-hashes.txt`: source/core hashes extracted from the original pre-run manifest.
- `intended-evidence-files.txt` and `evidence-manifest.json`: the exact compact payload and its SHA-256 hashes.

The raw 150 KiB server log, 6 MiB kernel snapshot, compiler caches and model/Inductor artifacts are intentionally excluded. No generation text or prompts are retained.
