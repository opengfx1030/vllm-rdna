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
# (name -> {dir, lib_sha256, rocblas, rocm, torch, rows_per_rank, status, auto}).
# Solver IDs are build-specific, so rows are only ever used for the rocBLAS
# build that produced them.
#
# Selection:
#   * explicit (third argument or $TUNABLEOP_PROFILE): use that profile and
#     validate its lib_sha256 against the loaded library. A mismatch makes the
#     launch fail (return 1) unless TUNABLEOP_ALLOW_MISMATCH=1, which warns
#     hard and proceeds anyway.
#   * preferred ($TUNABLEOP_PREFER, a recipe's soft choice): use that profile
#     when it is registered for the loaded library; otherwise warn and fall
#     through to the default. Never fails a launch.
#   * default: auto-select the first profile whose lib_sha256[:12] equals the
#     loaded library's sha256[:12] and that is not registered with
#     "auto": false (an opt-in profile only a preference or an explicit name
#     selects, e.g. a model's rows on top of the canonical ones).
#   * no registered match: a legacy <tunableop dir>/rocblas-<sha12>/ folder is
#     still honoured, then the per-user fallback
#     $HOME/.cache/tunableop/tunableop_results.csv (created).
#
# The helper logs the rows each rank file holds (ranks 0..TUNABLEOP_RANKS-1, 4
# when unset): a worker on device ordinal r reads tunableop_results<r>.csv.
#
# Lookup only by default (PYTORCH_TUNABLEOP_TUNING=0); set TUNING=1 to re-tune
# shapes missing from the table. Rows are never written to /tmp or the run CWD.
#
# configure_v620_tunableop keeps the historical strict behaviour (lookup is
# disabled unless this build's four rank files exist; no per-user fallback).

# Print "<name>\t<dir>\t<lib_sha256>\t<auto>" for every profile in the registry
# (auto is 0 for a profile registered with "auto": false, else 1).
# Empty (and non-zero) when the registry is absent or unparsable.
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
        print("%s\t%s\t%s\t%d" % (name, d, s, p.get("auto", True) is not False))' "$registry" 2>/dev/null
    elif command -v jq >/dev/null 2>&1; then
        jq -r '.profiles // {} | to_entries[]
               | select(.value.dir and .value.lib_sha256)
               | "\(.key)\t\(.value.dir)\t\(.value.lib_sha256)\t\(if .value.auto == false then 0 else 1 end)"' \
            "$registry" 2>/dev/null
    fi
}

# _tunableop_pick <registry> <library sha256[:12]> [requested] [preferred]
# Print "<mode>\t<name>\t<dir>\t<lib_sha256>" for the profile a launch selects
# (nothing when none applies). Modes: explicit (the named profile, hash not
# checked here), preferred (only when its lib_sha256[:12] equals the library's),
# auto (first profile without "auto": false that matches the library).
_tunableop_pick() {
    local registry=$1 library_id=$2 requested=${3:-} preferred=${4:-} picked=""
    if [[ -n $requested ]]; then
        _tunableop_profiles "$registry" |
            awk -F'\t' -v n="$requested" \
                '$1 == n { print "explicit\t" $1 "\t" $2 "\t" $3; exit }'
        return 0
    fi
    if [[ -n $preferred ]]; then
        picked=$(_tunableop_profiles "$registry" |
            awk -F'\t' -v n="$preferred" -v h="$library_id" \
                '$1 == n && substr($3, 1, 12) == h { print "preferred\t" $1 "\t" $2 "\t" $3; exit }')
    fi
    if [[ -z $picked ]]; then
        picked=$(_tunableop_profiles "$registry" |
            awk -F'\t' -v h="$library_id" \
                '$4 != "0" && substr($3, 1, 12) == h { print "auto\t" $1 "\t" $2 "\t" $3; exit }')
    fi
    [[ -z $picked ]] || printf '%s\n' "$picked"
}

# Rows (lines that are not Validator lines) of one rank file; 0 when unreadable.
_tunableop_row_count() {
    awk '!/^Validator/ && NF { n++ } END { print n + 0 }' "$1" 2>/dev/null || printf '0\n'
}

configure_tunableop() {
    local library=${1:-} rows_root=${2:-} requested=${3:-${TUNABLEOP_PROFILE:-}}
    local preferred=${TUNABLEOP_PREFER:-}
    local library_id selected_name="" mode="" rows_dir profile_line tune_state canonical
    local profile_dir profile_sha ranks=${TUNABLEOP_RANKS:-4} rank counts=""
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

    profile_line=$(_tunableop_pick "$rows_root/profiles.json" "$library_id" "$requested" "$preferred")
    if [[ -n $requested ]]; then
        if [[ -z $profile_line ]]; then
            printf 'ERROR: TunableOp profile "%s" is not registered in %s/profiles.json; refusing to start.\n' \
                "$requested" "$rows_root" >&2
            return 1
        fi
        IFS=$'\t' read -r mode selected_name profile_dir profile_sha <<< "$profile_line"
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
        if [[ -n $preferred && $profile_line != preferred$'\t'* ]]; then
            printf 'WARNING: preferred TunableOp profile "%s" (TUNABLEOP_PREFER) is not registered for rocBLAS build %s in %s/profiles.json; auto-selecting.\n' \
                "$preferred" "$library_id" "$rows_root" >&2
        fi
        if [[ -n $profile_line ]]; then
            IFS=$'\t' read -r mode selected_name profile_dir profile_sha <<< "$profile_line"
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

    # Ranks 4 and up (TP*PP > 4: TP8, TP4xPP2 stage 1) have their own rank files. torch opens
    # tunableop_results<device ordinal>.csv, and a worker whose file is missing runs the default
    # rocBLAS heuristic without a word (only PYTORCH_TUNABLEOP_VERBOSE=1 prints "could not open").
    local missing_high=""
    for ((rank = 4; rank < ranks; rank++)); do
        [[ -s $rows_dir/tunableop_results$rank.csv ]] || missing_high+=" $rank"
    done
    if [[ -n $missing_high ]]; then
        printf 'WARNING: TunableOp rows missing for rank(s)%s at %s: those workers run the default rocBLAS heuristic.\n' \
            "$missing_high" "$rows_dir" >&2
    fi

    tune_state=off
    [[ $PYTORCH_TUNABLEOP_TUNING == 1 ]] && tune_state=enabled
    export PYTORCH_TUNABLEOP_FILENAME=$rows_dir/tunableop_results.csv
    export PYTORCH_TUNABLEOP_ENABLED=1
    if [[ -n $selected_name ]]; then
        case $mode in
            explicit)
                printf 'TunableOp profile %s selected for rocBLAS build %s (rows: %s); tuning %s.\n' \
                    "$selected_name" "$library_id" "$rows_dir" "$tune_state" >&2
                ;;
            preferred)
                printf 'TunableOp profile %s preferred (TUNABLEOP_PREFER) for rocBLAS build %s (rows: %s); tuning %s.\n' \
                    "$selected_name" "$library_id" "$rows_dir" "$tune_state" >&2
                ;;
            *)
                printf 'TunableOp profile %s auto-selected for rocBLAS build %s.\n' \
                    "$selected_name" "$library_id" >&2
                printf 'TunableOp lookup enabled for profile %s (rocBLAS %s; rows: %s); tuning %s (set PYTORCH_TUNABLEOP_TUNING=1 to re-tune missing shapes).\n' \
                    "$selected_name" "$library_id" "$rows_dir" "$tune_state" >&2
                ;;
        esac
    else
        printf 'TunableOp lookup enabled for rocBLAS build %s (rows: %s); tuning %s.\n' \
            "$library_id" "$rows_dir" "$tune_state" >&2
    fi
    for ((rank = 0; rank < ranks; rank++)); do
        counts+=" $rank=$(_tunableop_row_count "$rows_dir/tunableop_results$rank.csv")"
    done
    printf 'TunableOp rows per rank (a worker on device ordinal r reads tunableop_results<r>.csv):%s\n' \
        "$counts" >&2
}

# Historical V620 entry point: same as the generic helper but strict about the
# hash-matched rank files (no per-user fallback). Callers unchanged.
configure_v620_tunableop() {
    TUNEOP_REQUIRE_HASH_ROWS=1 configure_tunableop "$@"
}
