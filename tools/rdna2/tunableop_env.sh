#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Build-keyed TunableOp helper. Adapted from leapdragon c41f8f4c8.
# Solution IDs belong to a rocBLAS build, not just a version number.
#
# This is the tools/rdna2 mirror of tools/rdna2_028/tunableop_env.sh. The
# generic entry point is `configure_tunableop <librocblas.so.5> <rows root>`;
# rows are read from <rows root>/rocblas-<sha256(lib)[:12]>/ and never written
# to /tmp or the run CWD. If the build has no rows the per-user path
# $HOME/.cache/tunableop/tunableop_results.csv is used instead.
#
# configure_v620_tunableop keeps its historical strict behaviour (lookup is
# disabled unless this build's four rank files exist) so the V620 coverage
# checks stay meaningful; it is implemented on top of the generic helper.
configure_tunableop() {
    local library=${1:-} rows_root=${2:-} library_id rows_dir rank canonical tune_state
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
    rows_dir=$rows_root/rocblas-$library_id
    for rank in 0 1 2 3; do
        if [[ ! -s $rows_dir/tunableop_results$rank.csv ]]; then
            if [[ ${TUNEOP_REQUIRE_HASH_ROWS:-0} == 1 ]]; then
                printf 'WARNING: TunableOp disabled: missing rank %s rows for rocBLAS %s at %s; using default FP16 algorithms.\n' "$rank" "$library_id" "$rows_dir" >&2
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
    printf 'TunableOp lookup enabled for rocBLAS build %s (rows: %s); tuning %s.\n' \
        "$library_id" "$rows_dir" "$tune_state" >&2
}

# Historical V620 entry point: same as the generic helper but strict about the
# hash-matched rank files (no per-user fallback). Callers unchanged.
configure_v620_tunableop() {
    TUNEOP_REQUIRE_HASH_ROWS=1 configure_tunableop "$@"
}
