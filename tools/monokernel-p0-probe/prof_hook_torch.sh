#!/usr/bin/env bash
# P0(b) LOAD_HOOK (torch profiler variant, serve_p0b.sh PROF=torch): profile a
# short window of steady decode at c=1 and c=8 via /start_profile and
# /stop_profile. Requests start first so the window holds decode steps only.
# Exported by the harness: URL PORT SERVED MODEL OUT. Must end in PASS/FAIL.
set -uo pipefail
BASE=http://127.0.0.1:$PORT
TOK=${PROF_TOKENS:-300}
LEAD_S=${PROF_LEAD_S:-2}       # past prefill
WIN_S=${PROF_WIN_S:-1.0}       # profiled decode window
PROMPT_WORDS=${PROF_PROMPT_WORDS:-700}   # ~1k prompt tokens
D=$OUT/torchprof
mkdir -p "$D"
prompt=$(python3 -c "print(' '.join(['The quick brown fox jumps over the lazy dog.']*($PROMPT_WORDS//9)))")
req() {
    python3 - "$URL" "$SERVED" "$1 $prompt" "$TOK" <<'PY'
import json, sys, urllib.request
url, model, prompt, tok = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
body = json.dumps({"model": model, "prompt": prompt, "max_tokens": tok,
                   "temperature": 0, "ignore_eos": True}).encode()
r = urllib.request.urlopen(urllib.request.Request(url, body, {"Content-Type": "application/json"}), timeout=600)
print(json.load(r)["usage"])
PY
}
ok=1
for conc in 1 8; do
    pids=()
    for i in $(seq 1 "$conc"); do req "c${conc}r${i}-$RANDOM" > "$D/req-c$conc-$i.txt" 2>&1 & pids+=($!); done
    sleep "$LEAD_S"
    echo "== c=$conc start_profile $(date +%s.%N)"
    curl -sf -m 180 -X POST "$BASE/start_profile" || { echo "start_profile failed"; ok=0; }
    sleep "$WIN_S"
    curl -sf -m 180 -X POST "$BASE/stop_profile" || { echo "stop_profile failed"; ok=0; }
    echo "== c=$conc stop_profile $(date +%s.%N)"
    for p in "${pids[@]}"; do wait "$p" || ok=0; done
    cat "$D"/req-c$conc-*.txt
    sleep 20   # let the workers flush the traces
done
(( ok )) && echo PASS || echo FAIL
