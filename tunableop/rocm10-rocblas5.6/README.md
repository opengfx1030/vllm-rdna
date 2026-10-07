# TunableOp profile — rocm10-rocblas5.6 (foreign build, thin)

This is a **first-class named profile** alongside `../rocm7.14-rocblas5.5/`, but
it is **thin and unqualified**: 70 rows per rank for a build that has not been
exercised on the fork's serving matrix. It was imported from PR #5 (commit
`ca83d922a`) as a reference copy of the Leapdragon/George V620 table. The helper
will select it when the loaded build matches, but before serving on it, run its
own capture/tune/curate campaign.

## Identity

| | this profile | canonical `../rocm7.14-rocblas5.5/` |
|---|---|---|
| profile name | `rocm10-rocblas5.6` | `rocm7.14-rocblas5.5` |
| rocBLAS build | `5.6.0.8d1ae90e` | `5.5.0.cd957402` |
| librocblas sha256[:12] | `c27e2252cc7a` | `f30bb442e9b5` |
| torch | `2.13.0+rocm10.0.0` | `2.12.0+rocm7.14.0` |
| HIP / ROCm | `7.15.26333` | `7.14.60850` |
| venv | a venv loading rocBLAS `5.6.0.8d1ae90e` | a venv loading rocBLAS `5.5.0.cd957402` |
| rows / rank | 70 | 783 |
| status | **thin, needs its own capture campaign** | canonical |

TunableOp solution IDs are **build-specific**: these solver numbers are only
valid for the exact `librocblas.so.5` that generated them. They are never folded
into the canonical profile.

## How selection works

Both profiles are live. `configure_tunableop` (in
`tools/rdna2_028/tunableop_env.sh`, mirror `tools/rdna2/tunableop_env.sh`)
resolves a profile from `tunableop/profiles.json`:

* **auto (default):** pick the profile whose `lib_sha256[:12]` equals
  `sha256(loaded librocblas.so.5)[:12]`. On a ROCm 7.14 serving venv this never
  selects this profile.
* **explicit:** `TUNABLEOP_PROFILE=rocm10-rocblas5.6` forces it and validates the
  profile's `lib_sha256` against the loaded library. On a mismatch the helper
  **fails the launch** unless `TUNABLEOP_ALLOW_MISMATCH=1` (experiments only),
  which warns hard and proceeds.

Provenance is preserved verbatim in `provenance.json`.

## To qualify it

Capture/tune/curate against a venv that actually loads the `5.6.0.8d1ae90e`
build (same pipeline as the canonical profile: `tunableop_rows_pipeline.sh` →
`curate_tunableop_rows.py`), then update `rows_per_rank` and `status` in
`provenance.json` and `profiles.json`. Until then, serve the canonical profile.
