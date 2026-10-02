#!/usr/bin/env bash
# Cells-only runner: exercises an ALREADY-RUNNING vLLM server (no boot, no
# teardown) with the same warmup/coherence/cell protocol as im_ab2.sh.
#
#   ARM=patched PORT=18121 TAG=im2_patched bash im_cells.sh
#
# No /tmp.
set -uo pipefail
V=${V:-/home/chenco_adm/Apps/vllm/venv-7.14.0_0.28.0}
RUN=${RUN:-/home/chenco_adm/w4a8_runs/pr33_validate}
MODEL=${MODEL:-/home/chenco_adm/hfcache/hub/models--wtdcode--Qwen3.8-Flash-Next-AWQ-W4A16/snapshots/0939125b929543a783ce700c90e36dd1a575c00c}
PORT=${PORT:?PORT required}
ARM=${ARM:?ARM required}
TAG=${TAG:-im2_$ARM}
CELLS=${CELLS:-"1 1024 512 701|2 1024 512 702|3 1024 512 703|1 16384 1024 711|2 16384 1024 712|3 16384 1024 713|8 1024 512 708|8 16384 1024 718"}
D=$RUN/$TAG
mkdir -p "$D/cells"
log(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$D/driver.log"; }

curl -s --max-time 5 "http://127.0.0.1:$PORT/v1/models" | grep -q flash-next || { log "no server on $PORT"; exit 1; }
log "=== CELLS-ONLY arm=$ARM port=$PORT tag=$TAG ==="

for wl in "1024 64 991" "16384 64 992"; do set -- $wl
  ( cd "$D" && "$V/bin/python" -m vllm.entrypoints.cli.main bench serve --backend openai --endpoint /v1/completions --base-url "http://127.0.0.1:$PORT" --model "$MODEL" --served-model-name flash-next --dataset-name random --random-input-len "$1" --random-output-len "$2" --num-prompts 1 --max-concurrency 1 --ignore-eos --request-rate inf --seed "$3" --temperature 0 --save-result --result-dir "$D/warmup_$1" ) >"$D/warmup_$1.log" 2>&1
done
log "warmup done"

"$V/bin/python" - "$PORT" >"$D/coherence.txt" 2>&1 <<'PY'
import json,sys,urllib.request,re
port=sys.argv[1]
def ask(p,mt=24):
    body=json.dumps({"model":"flash-next","prompt":p,"max_tokens":mt,"temperature":0.0,"ignore_eos":True}).encode()
    r=urllib.request.Request(f"http://127.0.0.1:{port}/v1/completions",data=body,headers={"Content-Type":"application/json"})
    with urllib.request.urlopen(r,timeout=600) as resp: return json.load(resp)["choices"][0]["text"] or ""
def garbage(t):
    if not t.strip(): return "empty"
    if re.search(r'(.)\1{29,}', t): return "repeat>=30"
    if sum(1 for c in t if not c.isprintable() and c not in "\n\t") > len(t)*0.1: return "nonprintable"
    if t.count("!")>len(t)*0.5: return "bang-flood"
    return "ok"
ok=0; tot=0
for name,p,exp in [("france","The capital of France is","paris"),("math","2 + 2 =","4"),("gpu","A graphics processing unit is","gpu")]:
    tot+=1
    try: t=ask(p)
    except Exception as e: t=f"<ERR {e!r}>"
    g=garbage(t); good=(exp in t.lower()) and g=="ok"; ok+=good
    print(f"[{name}] good={good} garbage={g} exp={exp!r} | {t[:100]!r}")
print(f"COHERENCE {ok}/{tot}")
PY
log "coherence: $(grep -m1 COHERENCE "$D/coherence.txt")"

IFS='|' read -ra _cells <<<"$CELLS"
for cell in "${_cells[@]}"; do set -- $cell
  N=$1; IN=$2; OUTL=$3; SEED=$4
  CD="$D/cells/c${N}_${IN}"
  mkdir -p "$CD"
  ( cd "$D" && "$V/bin/python" -m vllm.entrypoints.cli.main bench serve --backend openai --endpoint /v1/completions --base-url "http://127.0.0.1:$PORT" --model "$MODEL" --served-model-name flash-next --dataset-name random --random-input-len "$IN" --random-output-len "$OUTL" --num-prompts "$N" --max-concurrency "$N" --ignore-eos --request-rate inf --seed "$SEED" --temperature 0 --save-result --result-dir "$CD" ) >"$CD.log" 2>&1
  log "cell c=$N in=$IN | $(grep -m1 'Output token throughput' "$CD.log" | tr -s ' ') | $(grep -m1 'Mean TTFT' "$CD.log" | tr -s ' ') | $(grep -m1 'Mean TPOT' "$CD.log" | tr -s ' ')"
done

{ echo "cell,arm,requests,output_tok_s,total_tok_s,mean_ttft_ms,mean_tpot_ms,median_itl_ms,p99_itl_ms"
  for cell in "${_cells[@]}"; do set -- $cell
    N=$1; IN=$2
    CD="$D/cells/c${N}_${IN}"
    f=$(ls "$CD"/*.json 2>/dev/null | head -1)
    "$V/bin/python" - "$N" "$IN" "$ARM" "$f" <<'PY'
import json,sys
n,inn,arm,f=sys.argv[1:5]
d={}
try: d=json.load(open(f))
except Exception: pass
def g(*ks):
  for k in ks:
    if k in d: return d[k]
  return ""
print(f"c{n}_{inn},{arm},{g('completed','successful_requests')},{g('output_throughput')},{g('total_token_throughput')},{g('mean_ttft_ms','mean_ttft')},{g('mean_tpot_ms','mean_tpot')},{g('median_itl_ms')},{g('p99_itl_ms')}")
PY
  done; } >"$D/summary.csv" 2>"$D/summary.err"
log "=== done, summary: $D/summary.csv ==="
cat "$D/summary.csv" >>"$D/driver.log"
