#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# LOAD_HOOK of the G0a driver: the per-arm workload that runs after the server is up.
# serve_validate.sh exports URL PORT SERVED MODEL OUT; the driver exports ARM R SPIKE VENV.
# The output must end in PASS or FAIL (serve_validate.sh greps it).
set -uo pipefail
: "${ARM:?}" "${R:?}" "${SPIKE:?}" "${VENV:?}" "${URL:?}" "${SERVED:?}" "${OUT:?}"
URL=${URL%/v1/completions} # serve_validate.sh exports the completions endpoint; g0a_requests.py takes the server base
PY=$VENV/bin/python
RQ=$SPIKE/g0a_requests.py
PROMPTS=$R/prompts.json
STANDIN_PORT=${STANDIN_PORT:-5601}
rc=0
mkdir -p "$OUT/stats"

case $ARM in
    base_ds)
        "$PY" "$RQ" base --url "$URL" --served "$SERVED" --prompts "$PROMPTS" --out "$OUT/base.json" --repeat 2 || rc=1
        "$PY" "$RQ" compare --ref "$OUT/base.json" --test "$OUT/base.json" --aa --out "$OUT/aa.json" || rc=1
        ;;
    base_sd)
        "$PY" "$RQ" base --url "$URL" --served "$SERVED" --prompts "$PROMPTS" --out "$OUT/base.json" --repeat 1 || rc=1
        if [[ -s $R/arms/base_ds/base.json ]]; then
            "$PY" "$RQ" compare --ref "$R/arms/base_ds/base.json" --test "$OUT/base.json" --out "$OUT/vs_base_ds.json" || rc=1
        fi
        ;;
    producer | producer_real)
        store=$R/store
        [[ $ARM == producer_real ]] && store=$R/store_real
        mkdir -p "$store"
        "$PY" "$RQ" prefill --url "$URL" --served "$SERVED" --prompts "$PROMPTS" --out "$OUT/prefill.json" --store "$store" || rc=1
        ;;
    consumer | consumer_skip)
        "$PY" "$SPIKE/g0a_handshake_server.py" serve --store "$R/store" --port "$STANDIN_PORT" --max-seconds 7200 \
            > "$OUT/standin.log" 2>&1 &
        standin=$!
        sleep 3
        kill -0 "$standin" 2> /dev/null || { echo "stand-in handshake server died:"; cat "$OUT/standin.log"; rc=1; }
        if ((rc == 0)); then
            "$PY" "$RQ" decode --url "$URL" --served "$SERVED" --prompts "$PROMPTS" --pjson "$R/arms/producer/prefill.json" \
                --stats "$OUT/stats" --out "$OUT/decode.json" || rc=1
            "$PY" "$RQ" compare --ref "$R/arms/base_ds/base.json" --test "$OUT/decode.json" --out "$OUT/vs_base_ds.json" || rc=1
        fi
        kill "$standin" 2> /dev/null
        wait "$standin" 2> /dev/null
        ;;
    *)
        echo "unknown arm $ARM"
        rc=1
        ;;
esac

# Ablation arms are expected to diverge; their verdict is the comparison text, not the exit.
if [[ $ARM == consumer_skip ]]; then
    echo "ablation arm: divergence is the expected result (see COMPARE lines above)"
    echo PASS
    exit 0
fi
if ((rc == 0)); then echo PASS; else echo FAIL; fi
exit "$rc"
