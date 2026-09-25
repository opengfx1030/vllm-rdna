# V620 FP16 TunableOp rows

These rows reuse Leapdragon's build-specific, lookup-only TunableOp approach.
They were generated with the matching ROCm wheel SDK on four V620s. They do not
quantize weights or activations. Solution IDs must never be reused across a
different rocBLAS build, even when version strings match.

## Qualified build

`rocblas-c27e2252cc7a` contains 70 FP16 dense matrix shapes for 800, 1,024,
1,600, 2,048, 2,400, 3,072, 3,200, 4,000, 4,096, and 8,192 input rows. All shapes
passed independent FP32 comparisons on all four GPUs. `provenance.json` records
the full library hash and package versions.
Each rank uses an identical copy of the qualified rows.

The launcher checks the library hash and all four rank files. If the matching
table is unavailable, it logs a warning and uses default FP16 algorithms instead
of loading incompatible solver IDs. Missing aligned chunk shapes also produce
an explicit warning; normal serving continues. Online tuning stays disabled.
`check_v620_tuning.py --strict` makes incomplete coverage fail a release check;
it reads the final cache/batch CLI overrides. Runtime-generated rows under the testing directory take
precedence over these bundled rows; `V620_TUNABLEOP_ROOT` selects an explicit root.

```bash
V620_MM_LIMIT='{"image":4,"video":1}' \
V620_MTP_TOKENS=2 \
VLLM_RDNA_AR=1 VLLM_RDNA_AR_MAX_KB=64 HSA_FORCE_FINE_GRAIN_PCIE=1 \
V620_TUNABLEOP=1 \
V620_ROCBLAS_LIBRARY=/path/to/site-packages/_rocm_sdk_libraries/lib/librocblas.so.5 \
bash tools/rdna2/serve_v620_candidate.sh --max-num-batched-tokens 4096
```

## Cache-aligned prefill

With automatic block sizing, Flash-Next TP4 conversation caching aligns
intermediate chunk ends to an 800-token recurrent-state grid. With a 4,096-token scheduling budget, ordinary
chunks therefore contain 4,000 tokens. Exact-shape TunableOp lookup cannot use
4,096-token rows for these chunks. The table now includes the five multiples of
800 up to 4,000, retaining every previously qualified entry unchanged.

These added entries were generated with `tune_v620_fp16.py --batch-tokens 4000
800 1600 2400 3200`, merged with the existing entries for the same library hash,
and replayed with `qualify_v620_fp16.py` on all four V620s. The 3,072-token entries
were generated and qualified in the same way.

Use `--block-size 1024` with the 4,096-token MTP2 scheduling budget to retain
4,096-token ordinary chunks. The 1,024/2,048/3,072 rows cover intermediate stops
on this grid. The original 800-token automatic grid remains covered by the
additional 800/1,600/2,400/3,200/4,000 entries. Keep cache-aligned scheduling
enabled; disabling it breaks reusable conversation state.

The coverage check covers the known seven dense projection dimensions and aligned
chunks. Arbitrary prompt tails, mixed batches, or model/runner changes can still
introduce other shapes. Re-run the cold-context performance and prefix-reuse
checks before promoting any later optimization. Retain the previous release's
source, environment, tuning files and launch command for rollback.

An 8,192-token budget on the 1,024 grid additionally requires qualifying 5,120,
6,144 and 7,168-token rows; the older standalone 8,192 entries alone are not full
coverage. A different rocBLAS hash requires a separately qualified table.

## Earlier full-model validation

Tested with the existing Intel AutoRound INT4 expert checkpoint, FP16 dense
weights, original BF16 PLE in CPU RAM, TP4/EP4, and MTP disabled. Dense INT8 shadows
were disabled. The configured context was 262,144 tokens, with 4 GiB of KV memory
per GPU. This validation covered 16k and 32k benchmark tiers.

The unmodified `llm-context-bench` runner at
`92286b24065565f4929e78c45f776029480e9939` used its locked sampling and requested
1,024 output tokens. An explicit 11% input-size tolerance accommodates this
tokenizer; actual prompt counts are shown below.

| Workload | Actual prompt tokens | 4k batch prefill tok/s | 8k batch prefill tok/s | 8k batch decode tok/s |
| --- | ---: | ---: | ---: | ---: |
| Prose 16k | 16,750 | 1,400.49 | 1,418.08 | 42.12 |
| Prose 32k | 33,455 | 1,442.87 | 1,458.51 | 42.01 |
| Code 16k | 18,063 | 1,443.85 | 1,470.56 | 42.05 |
| Code 32k | 36,135 | 1,441.24 | 1,467.69 | 41.98 |

All eight performance trials completed 1,024 output tokens and passed the runner's
validity checks. Both configurations include the donor's independent `wvSplitK`
output allocation. The earlier shared-output implementation could overwrite a
retained projection result; its earlier apparent quality successes are not a
reliable correctness baseline.

Both corrected configurations passed the single/four-request smoke checks but
passed only two of four long-context quality cases. Both code cases returned
`billing_units: 59` instead of `61`. An untuned FP16 control with the same output
ownership fix produced the same failure. These results do not establish broad
model accuracy, and the candidate remains experimental pending that investigation.
The benchmark fixtures and scoring were not changed.

The 8k-batch run reached a healthy API in 248.30 seconds; the 4k-batch run took
243.26 seconds. Maximum worker model-loading time was 90.73 and 92.11 seconds,
respectively. These were successive starts with existing filesystem caches,
not a controlled cold-storage startup comparison. Model-loading memory was
19.18 GiB per GPU at batch 8k, plus the separately allocated 4 GiB KV cache.

## Regenerate for another rocBLAS build

Run only with the inference service stopped, using the isolated testing
environment and its matching SDK library path. The tuning script deliberately
removes the serving environment's TunableOp enable/tuning overrides before
importing Torch: those environment variables take precedence over the Python API.

```bash
.venv/bin/python tools/rdna2/tune_v620_fp16.py \
  --output-root /path/to/testing/tunableop \
  --batch-tokens 800 1024 1600 2048 2400 3072 3200 4000 4096 8192
.venv/bin/python tools/rdna2/qualify_v620_fp16.py /path/to/testing/tunableop/rocblas-HASH
```

The first command tunes GPU 0 and records solver rows after every measured shape.
The second replays them on all four cards against FP32 references before writing
the other three serving files. Run the full model's output checks and benchmark
after qualification. Tuning may choose a different floating-point reduction
order; retaining FP16 weights does not imply bit-identical outputs.
