#!/usr/bin/env bash
# P0(b) LOAD_HOOK for tools/rdna/port_v031/serve_validate.sh. The server runs
# with the venv's rocprofiler-sdk tool preloaded (serve_p0b.sh ROCPROF=1,
# kernel trace inside ROCPROF_COLLECTION_PERIOD). This hook drives steady
# decode traffic in two marked windows (c=1, then c=8) and writes the
# CLOCK_MONOTONIC window bounds (same clock as the rocprofv3 timestamps) to
# $OUT/rocprof/marks.txt for tools/monokernel-p0-probe/analyze_trace.py.
# Exported by the harness: URL PORT SERVED MODEL OUT. Must end in PASS/FAIL.
#
# Why not vLLM's --profiler-config torch: on this stack the workers' kineto
# session records CPU ops only (no kernel events even with
# torch_profiler_activities=["CPU","CUDA"]), while the same torch build records
# graph-replayed kernels in a standalone script. rocprofv3 --attach is not
# usable either (the venv ships no rocprof-attach; the /opt one finds no
# attach thread in the venv runtime).
set -uo pipefail
IDLE_S=${PROF_IDLE_S:-60}     # let the collection period open first
WIN_S=${PROF_WIN_S:-45}
TOK=${PROF_TOKENS:-400}
PROMPT_WORDS=${PROF_PROMPT_WORDS:-700}   # ~1k prompt tokens
D=${PROF_OUT:-$OUT/rocprof}
mkdir -p "$D"
prompt=$(python3 -c "print(' '.join(['The quick brown fox jumps over the lazy dog.']*($PROMPT_WORDS//9)))")
mono() { python3 -c "import time;print(f'{time.clock_gettime(time.CLOCK_MONOTONIC):.6f}')"; }
loop() {  # $1 = tag, $2 = end (monotonic s): back-to-back requests until end
    python3 - "$URL" "$SERVED" "$1 $prompt" "$TOK" "$2" <<'PY'
import json, sys, time, urllib.request
url, model, prompt, tok, end = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), float(sys.argv[5])
n = 0
while time.clock_gettime(time.CLOCK_MONOTONIC) < end:
    body = json.dumps({"model": model, "prompt": f"{n} {prompt}", "max_tokens": tok,
                       "temperature": 0, "ignore_eos": True}).encode()
    r = urllib.request.urlopen(urllib.request.Request(url, body, {"Content-Type": "application/json"}), timeout=900)
    u = json.load(r)["usage"]; n += 1
print(f"{n} requests, last usage {u}")
PY
}
echo "idle ${IDLE_S}s"; sleep "$IDLE_S"
: > "$D/marks.txt"
ok=1
for conc in 1 8; do
    t0=$(mono); end=$(python3 -c "print($t0 + $WIN_S)")
    pids=()
    for i in $(seq 1 "$conc"); do loop "c${conc}w${i}" "$end" > "$D/req-c$conc-$i.txt" 2>&1 & pids+=($!); done
    for p in "${pids[@]}"; do wait "$p" || ok=0; done
    t1=$(mono)
    echo "c$conc $t0 $t1" | tee -a "$D/marks.txt"
    cat "$D"/req-c$conc-*.txt
done
(( ok )) && echo PASS || echo FAIL
