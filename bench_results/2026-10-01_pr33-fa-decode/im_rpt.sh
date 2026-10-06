#!/usr/bin/env bash
# Repeat runner for variance: same cell measured with several seeds, each into
# its own result dir, then a spread summary. Runs against an already-running
# server (no boot/teardown).
#
#   ARM=patched PORT=18123 TAG=rpt_patched bash im_rpt.sh
set -uo pipefail
V=${V:-/home/chenco_adm/Apps/vllm/venv-7.14.0_0.28.0}
RUN=${RUN:-/home/chenco_adm/w4a8_runs/pr33_validate}
MODEL=${MODEL:-/home/chenco_adm/hfcache/hub/models--wtdcode--Qwen3.8-Flash-Next-AWQ-W4A16/snapshots/0939125b929543a783ce700c90e36dd1a575c00c}
PORT=${PORT:?PORT required}
ARM=${ARM:?ARM required}
TAG=${TAG:-rpt_$ARM}
RUNS=${RUNS:-"1 1024 512 801|1 1024 512 802|1 1024 512 803|2 1024 512 811|2 1024 512 812|2 1024 512 813|1 16384 1024 821|1 16384 1024 822|1 16384 1024 823|2 16384 1024 831|2 16384 1024 832|2 16384 1024 833"}
D=$RUN/$TAG
mkdir -p "$D/cells"
log(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$D/driver.log"; }

curl -s --max-time 5 "http://127.0.0.1:$PORT/v1/models" | grep -q flash-next || { log "no server on $PORT"; exit 1; }
log "=== REPEAT arm=$ARM port=$PORT tag=$TAG ==="

IFS='|' read -ra _runs <<<"$RUNS"
for r in "${_runs[@]}"; do set -- $r
  N=$1; IN=$2; OUTL=$3; SEED=$4
  CD="$D/cells/c${N}_${IN}_s${SEED}"
  mkdir -p "$CD"
  ( cd "$D" && "$V/bin/python" -m vllm.entrypoints.cli.main bench serve --backend openai --endpoint /v1/completions --base-url "http://127.0.0.1:$PORT" --model "$MODEL" --served-model-name flash-next --dataset-name random --random-input-len "$IN" --random-output-len "$OUTL" --num-prompts "$N" --max-concurrency "$N" --ignore-eos --request-rate inf --seed "$SEED" --temperature 0 --save-result --result-dir "$CD" ) >"$CD.log" 2>&1
  log "run c=$N in=$IN s=$SEED | $(grep -m1 'Output token throughput' "$CD.log" | tr -s ' ')"
done

{ echo "cell,seed,arm,output_tok_s,mean_ttft_ms,mean_tpot_ms,median_itl_ms"
  for r in "${_runs[@]}"; do set -- $r
    N=$1; IN=$2; SEED=$4
    CD="$D/cells/c${N}_${IN}_s${SEED}"
    f=$(ls "$CD"/*.json 2>/dev/null | head -1)
    "$V/bin/python" - "$N" "$IN" "$SEED" "$ARM" "$f" <<'PY'
import json,sys
n,inn,seed,arm,f=sys.argv[1:6]
d={}
try: d=json.load(open(f))
except Exception: pass
def g(*ks):
  for k in ks:
    if k in d and d[k] not in (None,""): return d[k]
  return ""
print(f"c{n}_{inn},{seed},{arm},{g('output_throughput')},{g('mean_ttft_ms')},{g('mean_tpot_ms')},{g('median_itl_ms')}")
PY
  done; } >"$D/repeats.csv" 2>"$D/repeats.err"

"$V/bin/python" - "$D/repeats.csv" <<'PY'
import csv,sys,statistics as st
rows=list(csv.DictReader(open(sys.argv[1])))
groups={}
for r in rows:
    if r['output_tok_s']=="": continue
    groups.setdefault(r['cell'],[]).append(float(r['output_tok_s']))
print(f"{'cell':10} {'n':>2} {'mean':>8} {'min':>8} {'max':>8} {'spread%':>8}")
for c,v in sorted(groups.items()):
    m=st.mean(v); sp=100*(max(v)-min(v))/m
    print(f"{c:10} {len(v):>2} {m:8.2f} {min(v):8.2f} {max(v):8.2f} {sp:7.1f}%")
PY
