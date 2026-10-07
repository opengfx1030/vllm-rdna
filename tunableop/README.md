# V620 FP16 TunableOp profiles

Build-specific, **lookup-only** TunableOp solver rows for the gfx1030 serving
stack on 4× Radeon PRO V620. They were generated with the matching ROCm wheel
SDK. They do not quantize weights or activations. Solution IDs must never be
reused across a different rocBLAS build, even when version strings match.

Both shipped row sets are **first-class, named profiles** (nothing is archived):
one visible folder per rocBLAS build, selected at launch.

## Profiles

| Profile | rocBLAS | ROCm / torch | `librocblas.so.5` sha256[:12] | rows / rank | Status |
|---|---|---|---:|---:|---|
| `rocm7.14-rocblas5.5/` | `5.5.0.cd957402` | `7.14.60850` / `2.12.0+rocm7.14.0` | `f30bb442e9b5` | **783** | canonical (serving build, `venv-7.14.0_0.28.0`) |
| `rocm10-rocblas5.6/` | `5.6.0.8d1ae90e` | `7.15.26333` / `2.13.0+rocm10.0.0` | `c27e2252cc7a` | 70 | **thin, needs its own capture campaign** |

The canonical profile covers all three serving families:

* **Flash-Next** — `Qwen3.8-Flash-Next-AWQ-W4A16`, MTP=0 and MTP=2 (includes the
  M ≤ 8 decode shapes and the M = 16/24/32 MTP-verify shapes).
* **27B AWQ dense** — the dense projection shapes at the aligned prefill grid.
* **EXL3 27B** — `Qwen3.8-27B-exl3-3.00bpw` (K = 5120/1536/4352; 0 collisions
  with the Flash-Next K set).

Each profile carries a `provenance.json` (library hash, package versions, the
capture/tune/curate campaign, lookup-hit proof) and a `README.md` (per-campaign
deltas and regeneration recipe).

## Registry and selection

`profiles.json` is the registry: `name -> {rocblas, rocm, torch, lib_sha256,
dir, rows_per_rank, status}`. The helper
(`tools/rdna2_028/tunableop_env.sh`, mirror `tools/rdna2/tunableop_env.sh`)
resolves a profile from it:

```bash
source <tree>/tools/rdna2_028/tunableop_env.sh
configure_tunableop <librocblas.so.5 path> <tunableop dir> [profile]
```

* **auto (default, omitted/empty `[profile]`)** — pick the profile whose
  `lib_sha256[:12]` equals `sha256(loaded librocblas.so.5)[:12]`. This is the
  safe default: solver IDs are build-specific, so the rows must match the build
  that generated them. A legacy `rocblas-<hash>/` folder is still honoured if
  no registered profile matches, and the per-user path is the last resort.
* **explicit (`TUNABLEOP_PROFILE=<name>` or the third argument)** — force a
  named profile. The helper **validates its `lib_sha256` against the loaded
  library** and **fails the launch loudly on mismatch**. For experiments only,
  `TUNABLEOP_ALLOW_MISMATCH=1` downgrades the error to a hard warning and uses
  the rows anyway. A mismatched set can never run silently.

A healthy auto-selecting start logs:

```
TunableOp profile rocm7.14-rocblas5.5 auto-selected for rocBLAS build f30bb442e9b5.
TunableOp lookup enabled for profile rocm7.14-rocblas5.5 (rocBLAS f30bb442e9b5; rows: <tree>/tunableop/rocm7.14-rocblas5.5); tuning off ...
```

Lookup is read-only by default (`PYTORCH_TUNABLEOP_TUNING=0`).
`PYTORCH_TUNABLEOP_TUNING=1` re-tunes shapes missing from the table (writes into
the selected rows path; harvest into the profile directory when validated).

## Storage policy (mandatory)

* **One folder per rocBLAS build**, registered by name in `profiles.json`. The
  helper selects the folder that matches the *loaded* build; solver IDs from
  another build never validate.
* **Never** write rows to `/tmp` or the run CWD — both are wiped or vary between
  runs, which silently drops small-batch shapes back to rocBLAS heuristics.
* **Per-user fallback:** `$HOME/.cache/tunableop/tunableop_results.csv`. The
  helper uses it automatically when the selected build has no rows, and it is
  where online tuning writes.

All serve launchers under `scripts/` (through `scripts/rdna_launcher_common.sh`)
and the in-tree bench/capture drivers go through this helper (or the
`tools/rdna2/` mirror); a launcher with a `TUNABLEOP=0` opt-out sets
`PYTORCH_TUNABLEOP_ENABLED=0` and writes nothing.

## Cache-aligned prefill

With automatic block sizing, Flash-Next TP4 conversation caching aligns
intermediate chunk ends to an 800-token recurrent-state grid. With a 4,096-token
scheduling budget, ordinary chunks therefore contain 4,000 tokens. Exact-shape
lookup cannot use 4,096-token rows for these chunks; the table carries the five
multiples of 800 up to 4,000 (and 1,024/2,048/3,072 for the 1,024 grid).

Use `--block-size 1024` with the 4,096-token MTP2 scheduling budget to retain
4,096-token ordinary chunks. Keep cache-aligned scheduling enabled; disabling it
breaks reusable conversation state. An 8,192-token budget on the 1,024 grid
additionally needs 5,120/6,144/7,168 rows.

The coverage check covers the known dense projection dimensions and aligned
chunks. Arbitrary prompt tails, mixed batches, or model/runner changes can still
introduce other shapes. Re-run the cold-context performance and prefix-reuse
checks before promoting any later optimization, and retain the previous release's
source, environment, tuning files and launch command for rollback. A different
rocBLAS hash requires a separately qualified profile.

## Regenerate for another rocBLAS build

Run only with the inference service stopped, using the isolated testing
environment and its matching SDK library path. The tuning scripts deliberately
remove the serving environment's TunableOp enable/tuning overrides before
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
after qualification. Tuning may choose a different floating-point reduction order;
retaining FP16 weights does not imply bit-identical outputs.

For the current fork pipeline (capture → tune → curate → freeze + lookup-hit
proof) see `tools/rdna2_028/tunableop_rows_pipeline.sh`,
`tools/rdna2_028/curate_tunableop_rows.py` and the per-campaign sections in
`bench_results/`. Freeze the result into `tunableop/<profile-name>/`, then add
the profile to `profiles.json`.
