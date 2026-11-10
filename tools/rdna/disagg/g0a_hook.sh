#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# LOAD_HOOK of the G0a driver: the per-arm workload that runs after the server is up.
# serve_validate.sh exports URL PORT SERVED MODEL OUT; the driver exports ARM R SPIKE VENV.
# The output must end in PASS or FAIL (serve_validate.sh greps it).
# G0R_TRACE=0 disables the worker-side state tracer (g0r_trace.py); ONLY=<ids> limits the prompts.
set -uo pipefail
: "${ARM:?}" "${R:?}" "${SPIKE:?}" "${VENV:?}" "${URL:?}" "${SERVED:?}" "${OUT:?}"
URL=${URL%/v1/completions} # serve_validate.sh exports the completions endpoint; g0a_requests.py takes the server base
PY=$VENV/bin/python
RQ=$SPIKE/g0a_requests.py
PROMPTS=$R/prompts.json
STANDIN_PORT=${STANDIN_PORT:-5601}
rc=0
mkdir -p "$OUT/stats"
common=(--url "$URL" --served "$SERVED" --prompts "$PROMPTS")
[[ -n ${ONLY:-} ]] && common+=(--only "$ONLY")
[[ ${G0R_TRACE:-1} == 1 ]] && common+=(--trace-dir "$OUT/trace")

case $ARM in
    base_ds)
        "$PY" "$RQ" base "${common[@]}" --out "$OUT/base.json" --repeat 2 || rc=1
        "$PY" "$RQ" compare --ref "$OUT/base.json" --test "$OUT/base.json" --aa --out "$OUT/aa.json" || rc=1
        ;;
    base_sd)
        "$PY" "$RQ" base --url "$URL" --served "$SERVED" --prompts "$PROMPTS" --out "$OUT/base.json" --repeat 1 || rc=1
        if [[ -s $R/arms/base_ds/base.json ]]; then
            "$PY" "$RQ" compare --ref "$R/arms/base_ds/base.json" --test "$OUT/base.json" --out "$OUT/vs_base_ds.json" || rc=1
        fi
        ;;
    producer | producer_mono | producer_mono_tail | producer_real)
        store=$R/store${ARM#producer}
        mkdir -p "$store"
        "$PY" "$RQ" prefill "${common[@]}" --out "$OUT/prefill.json" --store "$store" || rc=1
        ;;
    consumer | consumer_mono | consumer_mono_tail | consumer_skip)
        # consumer<suffix> decodes from the state producer<suffix> dumped
        store=$R/store${ARM#consumer}
        producer=producer${ARM#consumer}
        if [[ $ARM == consumer_skip ]]; then
            store=$R/store
            producer=producer
        fi
        "$PY" "$SPIKE/g0a_handshake_server.py" serve --store "$store" --port "$STANDIN_PORT" --max-seconds 7200 \
            > "$OUT/standin.log" 2>&1 &
        standin=$!
        sleep 3
        kill -0 "$standin" 2> /dev/null || { echo "stand-in handshake server died:"; cat "$OUT/standin.log"; rc=1; }
        if ((rc == 0)); then
            "$PY" "$RQ" decode "${common[@]}" --pjson "$R/arms/$producer/prefill.json" \
                --stats "$OUT/stats" --out "$OUT/decode.json" || rc=1
            "$PY" "$RQ" compare --label "$ARM" --ref "$R/arms/base_ds/base.json" --test "$OUT/decode.json" \
                --out "$OUT/vs_base_ds.json" || rc=1
            # control, not a verdict: the baseline's own prefix-hit (tail recompute) run is the
            # reference that isolates the transfer from the decode-side tail recompute
            "$PY" "$RQ" compare --label "$ARM" --ref "$R/arms/base_ds/base.json" --ref-rep hit --test "$OUT/decode.json" \
                --out "$OUT/vs_base_ds_hit.json" || true
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
