#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# serve_validate.sh LOAD_HOOK of e_run.sh: optionally repeat the concurrency probe
# (E_CONC_REPEATS times, the acceptance of a new row set), then replay fixed prompts at
# c=1 (e_replay.py) and copy the counter dumps to $E_OUT/dump. serve_validate.sh exports
# URL PORT SERVED MODEL OUT; the caller exports E_GROUPS, E_STATS_DIR, E_OUT and
# optionally E_MAX_TOKENS, E_TEXT_FILE (the source text of ":txt" groups) and
# E_CONC_REPEATS. Ends with PASS or FAIL.
set -uo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PY=${VENV:?VENV is required}/bin/python
rc=0
for ((i = 1; i <= ${E_CONC_REPEATS:-0}; i++)); do
    if "$PY" "$HERE/../port_v031/concurrency_probe.py" --url "${URL:?}" --model "${SERVED:?}" \
        --verbose --save-results "${OUT:?}/concurrency-repeat-$i.json" \
        > "$OUT/concurrency-repeat-$i.txt" 2>&1; then
        echo "concurrency repeat $i: PASS"
    else
        echo "concurrency repeat $i: FAIL"
        rc=1
    fi
    grep -E "^FAIL|^WARN|RESULT" "$OUT/concurrency-repeat-$i.txt" | tail -8 | sed 's/^/    /'
done
args=(--base-url "http://127.0.0.1:${PORT:?}" --model "$SERVED" --tokenizer "${MODEL:?}"
    --groups "${E_GROUPS:?}" --max-tokens "${E_MAX_TOKENS:-4}"
    --stats-dir "${E_STATS_DIR:?}" --out "${E_OUT:?}")
[[ -n ${E_TEXT_FILE:-} ]] && args+=(--text-file "$E_TEXT_FILE")
"$PY" "$HERE/e_replay.py" "${args[@]}" || rc=1
((rc == 0)) && echo "PASS hook" || echo "FAIL hook"
exit "$rc"
