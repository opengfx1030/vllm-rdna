# Full-M Mono router progress (mixed outcomes)

This bundle preserves three separate results from isolated gfx1030 work. They do not qualify a production full-M kernel.

## Cooperative capability probe: pass, narrow scope

On HIP 6 (V620, gfx1030, wave32; boot `1e8b5da1-759f-4dbb-ae87-e58c2faaff48`), the runtime reported cooperative launch support. An isolated cooperative sentinel passed direct execution and two graph replays with changed inputs. Source hash and boot ID were unchanged, and the privileged journal checker reported zero appended records and no GPU fault. This confirms that the API/runtime capability is available for the tested sentinel. It does not prove full-grid admission, occupancy safety, timeout behavior, integration with the existing Mono kernel, or a performance benefit.

## Immutable-parameter prototype: CPU-only build/resource result

A matched O3 build was performed with no visible GPUs. For the inspected full-M kernel, private segment size fell from 1920 to 1312 bytes and SGPR spills from 43 to 5; VGPR count remained 128, VGPR spills fell from 204 to 200, and group segment size changed from 30792 to 30788 bytes. No GPU execution occurred, so correctness and latency remain unvalidated.

## Strict full-M router prototype: failed correctness gate

The isolated strict run on HIP 6 used actual native tensors and changed-input checks. M=25, 32, and 64 completed. At M=128, the changed-input case with seed 250252 failed the strict candidate-versus-native FFN output check at row 104, with maximum absolute error `0.121002197265625`; the residual tensor remained exact. The run stopped at that failure. This rejects the candidate in its current form; there is no performance result to report.

No production code, binary, raw kernel journal, or large tensor payload is included. Compact provenance pins source/binary hashes, boot ID, selected check outcomes, and resource summaries. The strict runtime emitted an unrelated Triton import warning before reaching the explicit correctness failure; the report is attributed to that failed strict output gate only.
