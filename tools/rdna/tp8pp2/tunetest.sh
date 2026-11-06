#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Which TunableOp rows does a process on device ordinal D load? One subprocess per
# ordinal, 8 visible devices (slots A+B), a TP4-shape GEMM that is in the shipped table
# and a TP8-only shape that is not (tunetest2.py). Lookup only (TUNING=0),
# RECORD_UNTUNED off (with RECORD_UNTUNED=1 torch 2.12 does not read the file at all:
# rows_loaded=0 on every ordinal).
#
#   bash tunetest.sh ROWS_DIR [ORDINALS]     ROWS_DIR = <tree>/tunableop/<profile>
set -uo pipefail
# shellcheck source=slots.sh
source "$(dirname "${BASH_SOURCE[0]}")/slots.sh"
ROWS=${1:?ROWS_DIR}
ORDS=${2:-"0 1 2 3 4 5 6 7"}
verify_slot AB || exit 1
HIP_VISIBLE_DEVICES=$(slot_hip AB)
export HIP_VISIBLE_DEVICES
export PYTORCH_TUNABLEOP_ENABLED=1 PYTORCH_TUNABLEOP_TUNING=0 PYTORCH_TUNABLEOP_HIPBLASLT_ENABLED=0
export TORCH_BLAS_PREFER_HIPBLASLT=0 PYTORCH_TUNABLEOP_VERBOSE=1
export PYTORCH_TUNABLEOP_FILENAME=$ROWS/tunableop_results.csv
unset PYTORCH_TUNABLEOP_RECORD_UNTUNED PYTORCH_TUNABLEOP_UNTUNED_FILENAME
echo "rows dir: $ROWS ($(find "$ROWS" -maxdepth 1 -name 'tunableop_results[0-9]*.csv' | wc -l) rank files)"
for d in $ORDS; do
    "$VENV/bin/python" "$H/tunetest2.py" "$d" 2>&1 | grep -E '^ordinal|could not open'
done
