# MonoKernel correctness and candidate-core evidence

This bundle records four distinct evidence sets: a disposable production-flags
O2 build and its runtime tests, an isolated quantized-target scratch check, a
combined-core dense native-reference check, and an isolated
dense-native-round-v2 scratch check. The scratch checks use separate binaries
and do not qualify the combined O2 core. No end-to-end serving or whole-model
qualification is included.

## Disposable production-flags O2 build

The local tracked K2 source SHA is
`bc8e35dd429fab98b55dbef2a87113794f1558f474bf1914fc98645262453dcf`.
Its exact generated HIP candidate SHA is
`898a2701d99bf5030ce50ab378976da4bbbb71c751d38576d88fba3d33a21383`. The
compiled disposable `_rocm_C.abi3.so` SHA is
`3f688006eec38adef6cf4d4546735f66b8892d8066387336a88ba6fe7b668bac`.
The build manifest pins the original O2 Ninja compile/link vectors and other
inputs. The candidate was not installed into the serving environment.

Explicit extraction from the compiled gfx1030 code object passed for all four
K2 specializations: VGPR 109, private segment 0 bytes, Wave32, LDS 25,280
bytes. The build wrapper's top-level exit status is `1` because its legacy
resource-file glob found no device object (`expected one device object: []`).
The separately run explicit object extractor passed on the compiled object and
produced the retained resource JSON and metadata. The build log records the
linked candidate hash above. Installed core and source hashes are identical
before and after this disposable build.

The builder's CPU-only mocked plan suite reports 8 tests passed, Ruff and
syntax passed, without compiler or GPU execution in that suite.

## Combined O2 core runtime tests

On boot `1e8b5da1-759f-4dbb-ae87-e58c2faaff48`, the old inline core
`25f3fc556ad5b77aabb0faa561d19a7fc95addfd8692bbda0ea26441d8d9913f` failed
all three newly added quantized router-boundary regression cases (M=3, 7, 24)
against the today's native module route IDs. The candidate O2 core above passed
the same three regression cases plus the existing 87-test native MonoKernel
suite: **90 passed**. The v3 runner reads the verified Ctrl layout and compares
exact route ID sets; the earlier v2 attempt is superseded because its test
helper/API path did not correctly read the registered state. The v3 before/after
source, binary, test, and boot pins match for both arms. This is kernel-test
evidence on one host/boot, not end-to-end model or serving qualification.

## Separate isolated correctness checks

**Quantized target:** the prebuilt isolated scratch validator reports PASS for
all 11 rows `1,2,3,4,5,6,7,8,12,16,24`, with strict route comparison, eager
checks, changed-input graph replays and fresh-eager self-checks. It loaded
scratch binary SHA `7a0538cfb63646af313c08c5275874316fd2077f4cca0e02099c75a163d9827e`,
not the combined O2 core above. The JSON pins its own native core/source,
fixture, boot, GPU and trace identity.

**Dense MTP on the combined O2 core:** actual native-reference checks report
PASS at rows `1,3,8,12,24` on the same boot and loaded candidate binary above.
They check native top-k IDs and published weights, output parity, two changed
input graph replays, and post-timing error/output/top-k state. The fixture is
synthetic FP16 dense-MTP data and the validator compares rank-local results;
this is a bounded kernel correctness check, not model serving qualification.
The runner, validator, fixture test source, per-row JSON/logs and before/after
pins are retained under `core-runtime/dense-native-reference/`.

The separate dense-native-round-v2 scratch diagnostics report PASS for all 11
rows `1,2,3,4,5,6,7,8,12,16,24`; two result folders contain rows from the same
boot and scratch binary. Dense-only and bundled-trace safety checks also report
PASS. The bundled trace includes the unique kernel trace SHA256 in
`dense/trace.sha256` and the compact kernel CSV. This remains experimental
scratch evidence, not combined production O2 runtime evidence.

These checks establish results only for their pinned fixtures, binaries and
recorded gates. The combined O2 binary passes the 90-test kernel suite and the
five-width dense native-reference suite; the old inline core reproduces three
quantized router-ID mismatches. They do not establish model serving behavior,
TP collective correctness, or full model correctness.

## File handling

`intended-evidence-files.txt` lists the complete review bundle.
`evidence-manifest.json` records hashes and sizes for each retained file.
Runner and build tool source snapshots have `.txt` suffixes to preserve the
exact executed bytes; restore the original suffix in a disposable directory
before execution. No compiled binaries, build caches, raw full runner logs, or
full HIP API trace are included.
