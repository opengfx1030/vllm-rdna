# Production BF16 core build and correctness gate

This records the rebuilt gfx1030 extension and postbuild tests before the BF16 core path was used for serving. It is a build and correctness gate, **not a serving speed result**. The remote editable build completed at 00:07:16 CEST on 2026-10-21. `sha256.core-bf16-binary` identifies `_rocm_C.abi3.so` as `c39006c814d32c95aac0388f12d0d26b7c6dc4201f071586a9ef4c8a69df8bd4`; the extracted gfx1030 code object hash and all 16 BF16 GEMV variants' wave32 metadata are retained separately. The only package-list change was the rebuilt vLLM wheel version label (`0.31.0+rocm714` to `0.31.0+rdna.rocm714`). The boot ID and source hashes are retained in this folder.

| Postbuild suite | Result | Evidence |
| --- | ---: | --- |
| BF16 kernel, dispatch and actual Inductor/CUDA graph route (M=3, 4, 24) | 105 passed, 1 skipped | `test-bf16-core-and-inductor-rerun.log` |
| FA writer layout, DFlash2 and profile tests | 161 passed | `test-fa-dflash-postbody.log` |
| LiLiCorr tests | 30 passed | `test-lilicorr-postbody.log` |

The fullgraph test requires the opaque `rocm_unquantized_gemm` op in the FX graph, confirms the BF16 kernel wrapper runs during capture, changes the activation before graph replay, and checks the replayed output against the changed-input reference. The final assertion accepts either FX's overload packet or its default overload. Older synthetic gfx11 tests explicitly mock `on_gfx10x=False`; this avoids accidentally taking the physical gfx1030 branch when using CPU fixtures. The initial pre-fix log is kept outside this compact folder at `/private/tmp/dflash2-2026-10-20-evidence/headopt-candidate-build/postbuild/test-bf16-core-and-inductor.log`; the passing rerun is the qualification result.

`provenance.csv` hashes the copied logs and manifests. The large disassembly dump is retained in the local raw artifact directory; its extracted wave32 summary and code-object hash are included here. Numerical, graph and architecture gates passed. Same-boot DFlash2 and native MTP2 serving checks have since completed; see `UNPROFILED_LONG_INTERMEDIATE.json` and `MTP2_SURFACE_AUDIT.md`. The DFlash2 objective remains unqualified because the measured c8 result is tied to native and c1 is within observed run variability.
