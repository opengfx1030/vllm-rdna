# Flash-Next fastest validated path (4× V620, gfx1030) — 2026-09-27

The exact configuration behind the numbers below, in reproduction order.

## Prerequisites

- Fork branch `rdna_extras` at `22d6e3462` or later (two-shot `rdna_ar`
  integration + the 20480 KiB gate default). It must include the
  draft-decode cudagraph fix `30632b2fa`; without it MTP≥1 loses 5–18 %.
- venv built as `venv-7.14.0_0.28.0` (torch 2.12.0+rocm7.14.0, hip 7.14.60850,
  rocblas 5.5.0.cd957402). The shared rows are keyed to `librocblas.so.5`
  sha256 `f30bb442e9b5…`.
- TunableOp rows: `tunableop/rocm7.14-rocblas5.5/tunableop_results{0..3}.csv`
  (783 rows/rank, committed to this repo). The launcher's helper auto-selects the
  profile from `tunableop/profiles.json` by the `librocblas.so.5` sha256 and
  enables **lookup-only** TunableOp (`TUNING` stays off).
- Model and PLE quant dir as in the launcher defaults (`MODEL`,
  `VLLM_PLE_QUANT_DIR` override).

## Commands

```bash
git clone https://github.com/opengfx1030/vllm-rdna.git && cd vllm-rdna
git checkout rdna_extras
MTP=2 bash tools/rdna/serve_gfx1030_flashnext_mtp.sh   # spec decode (16k+ workloads)
MTP=0 bash tools/rdna/serve_gfx1030_flashnext_mtp.sh   # plain decode (short prompts)
```

MTP=2 and the TunableOp rows are the launcher defaults. A healthy start logs
`TunableOp lookup enabled for rocBLAS build f30bb442e9b5; tuning stays off`
and `rdna_ar: one-shot all-reduce active (… max 20480 KB, oneshot 32 KB …)`.
If the `rdna_ar` line says `disabled`, the gate is too small (the two-shot
self-test fails below ~1 MiB) and the run silently falls back to RCCL, which
costs ~5–10 % decode — keep `VLLM_RDNA_AR_MAX_KB` at 20480.
A `WARNING` instead means the librocblas hash differs (rows are build-specific)
or the rows are missing — serving still works with default FP16 algorithms.

## Expected (4× V620, TP4, FP16, seed 12345, c=8)

| cell | MTP=0 | MTP=2 + rows |
| --- | ---: | ---: |
| 8×1k/512 | **166.3 tok/s** / 40.8 ms | 135.5 / 43.4 |
| 8×16k/1k | 67.3 / 75.6 | **73.8 / 64.9** |

Use MTP=2 for 16k+; MTP=0 for short-prompt throughput. Both cells were 8/8 with
coherent outputs (see `docs/rdna2/V620-*` and the companion repo's
`mtp-parity-quest-2026-09-27.md` for the per-step kernel budget).

## TunableOp storage policy

- **Never** store rows in `/tmp` or the run CWD. Both are wiped or vary between
  runs, which silently drops small-batch shapes back to rocBLAS heuristics (the
  cause of the lower c=1 numbers in the 2026-09-29 A/Bs).
- The fork's `tunableop/rocblas-<sha256[:12]>/` rows are the **shared source of
  truth** — they ship with the repo so every user gets the same tuned table.
- Per-user fallback: `~/.cache/tunableop/tunableop_results.csv` (used
  automatically when the loaded build has no rows; also where online tuning
  writes).
- Consumption is one line — the launchers already do it:
  `source tools/rdna2_028/tunableop_env.sh &&
   configure_tunableop "$ROCM_SDK_LIB/librocblas.so.5" "$PWD/tunableop"`.

## Verify

```bash
python -m vllm.entrypoints.cli.main bench serve --backend openai \
  --endpoint /v1/completions --base-url http://127.0.0.1:18096 \
  --model "$MODEL" --served-model-name flash-next --dataset-name random \
  --random-input-len 16384 --random-output-len 1024 \
  --num-prompts 8 --max-concurrency 8 --ignore-eos --request-rate inf --seed 12345
```

## Caveats

- Cold tuning is impractical online (>20 min of inline tuning per workload);
  that is why the rows are shipped rather than re-tuned at each start.
- The first requests after a fresh start JIT the QSA Triton kernels; that pass
  is slow (tens of seconds). Wait for it instead of treating it as a hang.
- Tuned kernels can change the floating-point reduction order: outputs are not
  bit-identical to an untuned run and greedy acceptance shifts slightly
  (1.81 vs 2.43 observed at 8×1k/512). Validation is 8/8 + coherent outputs.
- Solution IDs are build-specific. Never reuse the rows across a different
  rocBLAS build; the helper enforces this by hash.
