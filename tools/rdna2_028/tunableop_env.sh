#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Profile-aware TunableOp helper for the RDNA fork.
#
#   configure_tunableop <librocblas.so.5 path> <tunableop dir> [profile]
#
# Rows live in <tunableop dir>/<profile-dir>/tunableop_results{0..3}.csv, one
# folder per rocBLAS build, registered in <tunableop dir>/profiles.json
# (name -> {dir, lib_sha256, rocblas, rocm, torch, rows_per_rank, status}).
# Solver IDs are build-specific, so rows are only ever used for the rocBLAS
# build that produced them.
#
# Selection:
#   * explicit (third argument or $TUNABLEOP_PROFILE): use that profile and
#     validate its lib_sha256 against the loaded library. A mismatch makes the
#     launch fail (return 1) unless TUNABLEOP_ALLOW_MISMATCH=1, which warns
#     hard and proceeds anyway.
#   * default: auto-select the profile whose lib_sha256[:12] equals the loaded
#     library's sha256[:12].
#   * no registered match: a legacy <tunableop dir>/rocblas-<sha12>/ folder is
#     still honoured, then the per-user fallback
#     $HOME/.cache/tunableop/tunableop_results.csv (created).
#
# Lookup only by default (PYTORCH_TUNABLEOP_TUNING=0); set TUNING=1 to re-tune
# shapes missing from the table. Rows are never written to /tmp or the run CWD.
#
# configure_mtp_tunableop is an alias so existing callers do not break.
# configure_v620_tunableop keeps the historical strict behaviour (lookup is
# disabled unless this build's four rank files exist; no per-user fallback).

# Print "<name>\t<dir>\t<lib_sha256>" for every profile in the registry.
# Empty (and non-zero) when the registry is absent or unparseable.
_tunableop_profiles() {
    local registry=$1
    [ -r "$registry" ] || return 1
    if command -v python3 >/dev/null 2>&1; then
        python3 -c 'import json, sys
try:
    prof = json.load(open(sys.argv[1])).get("profiles", {})
except Exception:
    sys.exit(0)
for name in sorted(prof):
    p = prof[name] or {}
    d, s = p.get("dir"), p.get("lib_sha256")
    if d and s:
        print("%s\t%s\t%s" % (name, d, s))' "$registry" 2>/dev/null
    elif command -v jq >/dev/null 2>&1; then
        jq -r '.profiles // {} | to_entries[]
               | select(.value.dir and .value.lib_sha256)
               | "\(.key)\t\(.value.dir)\t\(.value.lib_sha256)"' \
            "$registry" 2>/dev/null
    fi
}

configure_tunableop() {
    local library=${1:-} rows_root=${2:-} requested=${3:-${TUNABLEOP_PROFILE:-}}
    local library_id selected_name="" rows_dir profile_line tune_state canonical
    local profile_dir profile_sha
    export PYTORCH_TUNABLEOP_HIPBLASLT_ENABLED=0
    export PYTORCH_TUNABLEOP_TUNING=${PYTORCH_TUNABLEOP_TUNING:-0}
    export PYTORCH_TUNABLEOP_ENABLED=0
    unset PYTORCH_TUNABLEOP_FILENAME
    canonical=${TUNEOP_PER_USER_DIR:-$HOME/.cache/tunableop}
    mkdir -p -- "$canonical" 2>/dev/null || true

    if [[ -z $library || ! -f $library ]]; then
        printf 'WARNING: TunableOp lookup disabled: rocBLAS library "%s" is unavailable; using default FP16 algorithms.\n' "$library" >&2
        return 0
    fi

    library_id=$(sha256sum -- "$library")
    library_id=${library_id:0:12}

    if [[ -n $requested ]]; then
        profile_line=$(_tunableop_profiles "$rows_root/profiles.json" |
            awk -F'\t' -v n="$requested" '$1 == n { print; exit }')
        if [[ -z $profile_line ]]; then
            printf 'ERROR: TunableOp profile "%s" is not registered in %s/profiles.json; refusing to start.\n' \
                "$requested" "$rows_root" >&2
            return 1
        fi
        selected_name=${profile_line%%$'\t'*}
        profile_dir=$(printf '%s' "$profile_line" | cut -f2)
        profile_sha=$(printf '%s' "$profile_line" | cut -f3)
        if [[ ${profile_sha:0:12} != "$library_id" ]]; then
            if [[ ${TUNABLEOP_ALLOW_MISMATCH:-0} == 1 ]]; then
                printf 'WARNING: TunableOp profile "%s" was tuned for rocBLAS %s but the loaded library is %s; TUNABLEOP_ALLOW_MISMATCH=1 set, using it anyway (results may be wrong or slow).\n' \
                    "$selected_name" "${profile_sha:0:12}" "$library_id" >&2
            else
                printf 'ERROR: TunableOp profile "%s" was tuned for rocBLAS %s, but the loaded librocblas.so.5 hashes to %s. Solver IDs are build-specific; refusing to start. Unset TUNABLEOP_PROFILE to auto-select, or set TUNABLEOP_ALLOW_MISMATCH=1 to override (experiments only).\n' \
                    "$selected_name" "${profile_sha:0:12}" "$library_id" >&2
                return 1
            fi
        fi
        rows_dir=$rows_root/$profile_dir
    else
        profile_line=$(_tunableop_profiles "$rows_root/profiles.json" |
            awk -F'\t' -v h="$library_id" 'substr($3, 1, 12) == h { print; exit }')
        if [[ -n $profile_line ]]; then
            selected_name=${profile_line%%$'\t'*}
            profile_dir=$(printf '%s' "$profile_line" | cut -f2)
            rows_dir=$rows_root/$profile_dir
        else
            # Legacy hash-named folder, then the per-user fallback below.
            rows_dir=$rows_root/rocblas-$library_id
        fi
    fi

    for rank in 0 1 2 3; do
        if [[ ! -s $rows_dir/tunableop_results$rank.csv ]]; then
            if [[ ${TUNEOP_REQUIRE_HASH_ROWS:-0} == 1 ]]; then
                printf 'WARNING: TunableOp disabled: missing rank %s rows for rocBLAS %s at %s; using default FP16 algorithms.\n' \
                    "$rank" "$library_id" "$rows_dir" >&2
                return 0
            fi
            tune_state=off
            [[ $PYTORCH_TUNABLEOP_TUNING == 1 ]] && tune_state=enabled
            printf 'WARNING: no fork rows for rocBLAS build %s (rank %s missing) at %s; falling back to per-user %s (tuning %s).\n' \
                "$library_id" "$rank" "$rows_dir" "$canonical/tunableop_results.csv" "$tune_state" >&2
            export PYTORCH_TUNABLEOP_FILENAME=$canonical/tunableop_results.csv
            export PYTORCH_TUNABLEOP_ENABLED=1
            return 0
        fi
    done

    tune_state=off
    [[ $PYTORCH_TUNABLEOP_TUNING == 1 ]] && tune_state=enabled
    export PYTORCH_TUNABLEOP_FILENAME=$rows_dir/tunableop_results.csv
    export PYTORCH_TUNABLEOP_ENABLED=1
    if [[ -n $selected_name ]]; then
        if [[ -n $requested ]]; then
            printf 'TunableOp profile %s selected for rocBLAS build %s (rows: %s); tuning %s.\n' \
                "$selected_name" "$library_id" "$rows_dir" "$tune_state" >&2
        else
            printf 'TunableOp profile %s auto-selected for rocBLAS build %s.\n' \
                "$selected_name" "$library_id" >&2
            printf 'TunableOp lookup enabled for profile %s (rocBLAS %s; rows: %s); tuning %s (set PYTORCH_TUNABLEOP_TUNING=1 to re-tune missing shapes).\n' \
                "$selected_name" "$library_id" "$rows_dir" "$tune_state" >&2
        fi
    else
        printf 'TunableOp lookup enabled for rocBLAS build %s (rows: %s); tuning %s.\n' \
            "$library_id" "$rows_dir" "$tune_state" >&2
    fi
}

# Alias kept for the MTP/Flash-Next launchers and other existing callers.
configure_mtp_tunableop() {
    configure_tunableop "$@"
}

# Historical V620 entry point: same as the generic helper but strict about the
# hash-matched rank files (no per-user fallback). Callers unchanged.
configure_v620_tunableop() {
    TUNEOP_REQUIRE_HASH_ROWS=1 configure_tunableop "$@"
}
