#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# LOAD_HOOK of scoreboard.sh for the TunableOp row gate: repeated runs of the
# deterministic concurrency probe (tools/rdna/port_v031/concurrency_probe.py) against
# the arm's running server, the protocol of docs/rdna2/tp8-pp2-flashnext-2026-11-06.md
# (a tuned row set is accepted only when no run flips a token that the rows-off arm
# keeps). serve_validate.sh exports URL PORT SERVED MODEL OUT to the hook; scoreboard.sh
# exports VENV.
#
#   QUALITY=0 CELLS=none CONC_REPEAT=10 CONC_REPEAT_NOEOS=6 \
#       LOAD_HOOK=$PWD/tools/rdna/rdna_hint_flex/conc_repeat_hook.sh \
#       bash tools/rdna/rdna_hint_flex/scoreboard.sh ARM [KEY=value ...]
#
#   CONC_REPEAT        runs with the probe's default --ignore-eos (every request 64 tokens),
#                      files $OUT/conc/run<N>.{txt,json}   [10]
#   CONC_REPEAT_NOEOS  runs with --no-ignore-eos (requests stop where serving would),
#                      files $OUT/conc/noeos<N>.{txt,json} [6]
#   CONC_PROBE_ARGS    extra probe arguments for both
#
# One line per run in $OUT/conc/runs.tsv (label, run, exit code, the probe's RESULT line;
# exit code 0 = the probe's own PASS). Tally and Fisher test across arms:
#   $VENV/bin/python tools/rdna/tp8pp2/conctally.py <OUT_ROOT> a=ARM_A b=ARM_B
set -uo pipefail
: "${OUT:?}" "${PORT:?}" "${SERVED:?}" "${VENV:?}"
H=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROBE=$H/../port_v031/concurrency_probe.py
mkdir -p "$OUT/conc"
: > "$OUT/conc/runs.tsv"
read -r -a extra <<< "${CONC_PROBE_ARGS:-}"
rc=0

# runs LABEL COUNT [probe args]
runs() {
    local label=$1 count=$2 i crc npass=0
    shift 2
    for ((i = 1; i <= count; i++)); do
        "$VENV/bin/python" "$PROBE" --url "http://127.0.0.1:$PORT/v1/completions" \
            --model "$SERVED" --verbose --save-results "$OUT/conc/$label$i.json" \
            "$@" > "$OUT/conc/$label$i.txt" 2>&1
        crc=$?
        printf '%s\t%s\t%s\t%s\n' "$label" "$i" "$crc" \
            "$(grep -aE '^RESULT' "$OUT/conc/$label$i.txt" | tail -1)" >> "$OUT/conc/runs.tsv"
        ((crc == 0)) && npass=$((npass + 1))
    done
    echo "[$(date +%T)] CONC $label: $npass of $count runs PASS (tally $OUT/conc/runs.tsv)"
    ((npass == count)) || rc=1
}

runs run "${CONC_REPEAT:-10}" "${extra[@]}"
runs noeos "${CONC_REPEAT_NOEOS:-6}" "${extra[@]}" --no-ignore-eos
grep -ahE '^FAIL' "$OUT"/conc/run[0-9]*.txt "$OUT"/conc/noeos[0-9]*.txt | head -20
if ((rc == 0)); then echo "PASS conc_repeat"; else echo "FAIL conc_repeat"; fi
exit "$rc"
