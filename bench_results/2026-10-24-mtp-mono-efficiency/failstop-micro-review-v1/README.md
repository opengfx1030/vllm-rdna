# MTP fail-stop health micro evidence

This bundle records a bounded TP4 health-collective microcheck, a disposable
two-unit core build/resource extraction, and a direct native-kernel test suite.
It does **not** qualify the full inference engine, a serving workload, or a
production deployment. The production source edits are tracked separately.

## TP4 health collective

`collective/summary.json` and `rank0.json`–`rank3.json` preserve the successful
attempt 5. The test used the existing `layout.workspace(device, 'mtp')` coarse
HSA allocation (5,138,176 bytes, `mem_flags=4`) and a separate ordinary Torch
allocation sentinel. All four workers mapped to HIP 6–9 on one host boot.

The checks passed for eager fault propagation from ranks 1 and 3, immutable
inputs, non-aliasing outputs, sticky workspace error state, prefix reset,
fallback zero, and returned-output lifetime. Captured graph replays with
changed fault inputs on ranks 1 and 3 returned the expected error; a cleared
input replay returned zero. `RDNA_AR` was enabled, but its int32 eligibility
was false; the run used the process-group/communicator fallback. These results
therefore do not demonstrate int32 support in RDNA_AR.

The healthy collective median was about **44.2 microseconds** per call. The
healthy collective plus asynchronous 4-byte device-to-host check was about
**135.7 microseconds** per call. Both are isolated microcheck timings, not
inference latency or a serving overhead estimate.

## Disposable core build and native tests

`core-build/` records a clean disposable core build and its object resource
metadata. The built extension SHA-256 was
`6facdbd60118fc14a5cf1ceb5c2494d38ed1be09f55e4636bb0f0825678e2eed`; it was
not installed into the serving environment. The modified MonoKernel object
used 109 VGPR, zero private bytes, and Wave32 for the four FFN variants. The
full kernel suite then passed **289 tests, skipped 48, with 14 warnings**.
Those direct tests cover kernels and collectives, not end-to-end engine safety
or model serving.

The source transform reproduced the base HIP bytes exactly before applying
the intended generated candidate. `inputs.before` and `inputs.after` document
the inputs and unchanged-object proof; object files, extension binaries, and
code images are deliberately omitted.

## Failed harness starts

The four earlier harness attempts failed before reaching the collective. The
short history in `harness-history/attempt-history.md` records each startup
assumption and correction. These failures are retained to explain why the
successful attempt used both preserved selector aliases and KFD queue-based
physical-device verification.

## Ordinal-to-KFD mapping check

Run `/private/tmp/mono-local-validation-venv/bin/python check_ordinal_mapping.py.txt`
to check the recorded ordinal mapping without importing Torch or initializing
a GPU. The test distinguishes logical visible indices 6–9 from KFD GPU IDs
56424, 18860, 17318, and 29156.

## Provenance and limits

The successful collective and build ran on host boot
`1e8b5da1-759f-4dbb-ae87-e58c2faaff48`. Source hashes, environment, worker rank
metadata, and input manifests are retained in the subdirectories. The
successful collective runner and launcher are snapshotted as `.txt` files so
repository hooks cannot rewrite executed evidence. No model output, raw host
journal, executable/object binary, or code image is included.
