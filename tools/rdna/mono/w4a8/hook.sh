#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# serve_validate.sh LOAD_HOOK for one mono + W4A8 arm: arm_driver.py measures the
# running server (eviction-checked perf cells, decode logprob parity, teacher-
# forced windows). serve_validate.sh exports PORT, SERVED, MODEL and OUT;
# slot.sh exports the MONO_* knobs below. Ends with PASS or FAIL.
set -uo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
: "${PORT:?}" "${SERVED:?}" "${MODEL:?}" "${OUT:?}"
PY=${PY:-$HOME/Apps/vllm/venv-7.14.0_0.32.0/bin/python}
exec "$PY" "$HERE/arm_driver.py" \
    --url "http://127.0.0.1:$PORT" --model "$SERVED" --tokenizer "$MODEL" \
    --out "$OUT" --py "$PY" --tp "${MONO_TP:-4}" \
    --phases "${MONO_PHASES:-perf,parity,topk}" --cells "${MONO_CELLS:-}" \
    --cell-tries "${MONO_CELL_TRIES:-3}" --windows "${MONO_WINDOWS:-}" \
    --topk-limit "${MONO_TOPK_LIMIT:-32}" --topk-repeat "${MONO_TOPK_REPEAT:-16}"
