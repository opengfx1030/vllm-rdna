#!/usr/bin/env bash
# PR33 in-model A/B: Flash-Next AWQ-W4A16, FA-RDNA2, TP=4 on GPUs 4-7.
# Boots one arm at a time with a per-arm cache root, runs the cells, tears down.
# No /tmp. Stop on new PCI SERR.
set -uo pipefail
V=/home/chenco_adm/Apps/vllm/venv-7.14.0_0.28.0
T=/home/chenco_adm/vllm-rdna-0.28.0
RUN=/home/chenco_adm/w4a8_runs/pr33_validate
MODEL=${MODEL:-/home/chenco_adm/hfcache/hub/models--wtdcode--Qwen3.8-Flash-Next-AWQ-W4A16/snapshots/0939125b929543a783ce700c90e36dd1a575c00c}
PLE=${PLE:-/home/chenco_adm/hfcache/hub/models--primitive-ai--Qwen3.8-Flash-Next-PLE-quant/snapshots/4f861b63f69e61bfc2e22130ec91ec67f03ec43e/ples_int4}
RECIPE=${RECIPE:-flashnext-mtp0}
PORT=${PORT:-18120}
ARMS=${ARMS:-"baseline patched"}
CELLS=${CELLS:-"1 1024 512 701|2 1024 512 702|3 1024 512 703|1 16384 1024 711|2 16384 1024 712|3 16384 1024 713|8 1024 512 708|8 16384 1024 718"}
DRIVER_LOG=$RUN/im_ab.log
log(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$DRIVER_LOG"; }

our_pids() {
  local p exe
  for p in $(pgrep -f "entrypoints.cli.main serve|entrypoints.openai.api_server|VLLM::Worker|VLLM::EngineCore|PleOffloadWorker" 2>/dev/null); do
    exe=$(readlink -f "/proc/$p/exe" 2>/dev/null || true)
    case "$exe" in */venv-7.14.0_0.28.0/bin/python*) echo "$p" ;; esac
  done
}
hygiene() {
  local n=0 left p
  for p in $(our_pids); do kill -TERM "$p" 2>/dev/null && n=$((n+1)); done
  [ "$n" -gt 0 ] && sleep 8
  for p in $(our_pids); do kill -KILL "$p" 2>/dev/null; done
  sleep 2
  left=$(our_pids | tr '\n' ' ')
  log "hygiene: terminated=$n remaining=[${left% }]"
  [ -z "${left// /}" ]
}
sel_count(){ timeout 10 sudo -n ipmitool sel list 2>/dev/null | grep -c 'PCI SERR'; }

teardown(){
  local CACHE=$1 p
  for p in $(pgrep -f "entrypoints.cli.main serve|entrypoints.openai.api_server|VLLM::Worker|VLLM::EngineCore|PleOffloadWorker" 2>/dev/null); do
    if [ -r "/proc/$p/environ" ] && tr '\0' '\n' <"/proc/$p/environ" 2>/dev/null | grep -q "^VLLM_CACHE_ROOT=$CACHE$"; then kill -TERM "$p" 2>/dev/null; fi
  done
  sleep 10
  for p in $(pgrep -f "entrypoints.cli.main serve|entrypoints.openai.api_server|VLLM::Worker|VLLM::EngineCore|PleOffloadWorker" 2>/dev/null); do
    if [ -r "/proc/$p/environ" ] && tr '\0' '\n' <"/proc/$p/environ" 2>/dev/null | grep -q "^VLLM_CACHE_ROOT=$CACHE$"; then kill -KILL "$p" 2>/dev/null; fi
  done
  sleep 3
}

run_arm(){
  local arm=$1
  local CACHE=$T/cache/ab-pr33-$arm
  local D=$RUN/im_$arm
  mkdir -p "$D/cells"
  cp "$RUN/_rocm_C.$arm.so" "$T/vllm/_rocm_C.abi3.so"
  local so; so=$(sha256sum "$T/vllm/_rocm_C.abi3.so" | cut -c1-16)
  log "=== ARM=$arm so=$so recipe=$RECIPE port=$PORT ==="
  hygiene || { log "hygiene failed"; return 1; }
  if ss -ltn 2>/dev/null | grep -q ":$PORT "; then log "port $PORT busy"; return 1; fi
  rm -rf "$CACHE"; mkdir -p "$CACHE/inductor" "$CACHE/extensions" "$CACHE/triton" "$CACHE/vllm"
  rm -f "$CACHE/rdna_ar_wedged"
  local S0; S0=$(sel_count); log "PCI-SERR before=$S0 uptime=$(uptime -p)"
  local T0; T0=$(date +%s)
  setsid nohup env \
    VENV="$V" VLLM_TREE="$T" MODEL="$MODEL" VLLM_PLE_QUANT_DIR="$PLE" \
    VLLM_CACHE_ROOT="$CACHE" TORCHINDUCTOR_CACHE_DIR="$CACHE/inductor" \
    TRITON_CACHE_DIR="$CACHE/triton" TORCH_EXTENSIONS_DIR="$CACHE/extensions" \
    HIP_VISIBLE_DEVICES=4,5,6,7 \
    bash "$T/scripts/serve_rdna.sh" RECIPE="$RECIPE" PORT="$PORT" \
    >"$D/serve.log" 2>&1 </dev/null &
  local ready=0 dead=0 i
  for i in $(seq 1 90); do
    sleep 10
    curl -s --max-time 4 "http://127.0.0.1:$PORT/v1/models" 2>/dev/null | grep -q flash-next && { ready=1; break; }
    if pgrep -f "[a]pi_server" >/dev/null; then dead=0; else dead=$((dead+1)); fi
    [ "$dead" -ge 3 ] && { log "server gone during boot"; break; }
  done
  local COLD=$(( $(date +%s) - T0 ))
  if [ "$ready" != 1 ]; then log "NOT READY after ${COLD}s"; tail -30 "$D/serve.log" >>"$DRIVER_LOG"; hygiene; return 1; fi
  log "READY t=${COLD}s"; echo "$COLD" >"$D/cold_seconds.txt"
  local S1; S1=$(sel_count); log "PCI-SERR after boot=$S1 uptime=$(uptime -p)"
  if [ "$S1" -gt "$S0" ]; then echo "STOP NEW PCI SERR" >"$RUN/IM_STOP"; hygiene; return 2; fi
  # Warmup (throwaway)
  local wl
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
  local cell
  for cell in "${_cells[@]}"; do set -- $cell
    local N=$1 IN=$2 OUTL=$3 SEED=$4
    local CD="$D/cells/c${N}_${IN}"
    mkdir -p "$CD"
    ( cd "$D" && "$V/bin/python" -m vllm.entrypoints.cli.main bench serve --backend openai --endpoint /v1/completions --base-url "http://127.0.0.1:$PORT" --model "$MODEL" --served-model-name flash-next --dataset-name random --random-input-len "$IN" --random-output-len "$OUTL" --num-prompts "$N" --max-concurrency "$N" --ignore-eos --request-rate inf --seed "$SEED" --temperature 0 --save-result --result-dir "$CD" ) >"$CD.log" 2>&1
    log "cell c=$N in=$IN | $(grep -m1 'Output token throughput' "$CD.log" | tr -s ' ') | $(grep -m1 'Mean TTFT' "$CD.log" | tr -s ' ') | $(grep -m1 'Mean TPOT' "$CD.log" | tr -s ' ')"
    local S2; S2=$(sel_count); if [ "$S2" -gt "$S0" ]; then echo "STOP NEW PCI SERR" >"$RUN/IM_STOP"; break; fi
  done
  { echo "cell,requests,output_tok_s,total_tok_s,mean_ttft_ms,mean_tpot_ms,median_itl_ms"
    for cell in "${_cells[@]}"; do set -- $cell
      local N=$1 IN=$2
      local CD="$D/cells/c${N}_${IN}"
      local f; f=$(ls "$CD"/*.json 2>/dev/null | head -1)
      "$V/bin/python" - "$N" "$IN" "$f" <<'PY'
import json,sys
n,inn,f=sys.argv[1],sys.argv[2],sys.argv[3]
d={}
try: d=json.load(open(f))
except Exception: pass
def g(*ks):
  for k in ks:
    if k in d: return d[k]
  return ""
print(f"c{n}_{inn},{g('completed','successful_requests')},{g('output_throughput')},{g('total_token_throughput')},{g('mean_ttft_ms','mean_ttft')},{g('mean_tpot_ms','mean_tpot')},{g('median_itl_ms')}")
PY
    done; } >"$D/summary.csv" 2>"$D/summary.err"
  teardown "$CACHE"; hygiene
  local S3; S3=$(sel_count); log "PCI-SERR after teardown=$S3 uptime=$(uptime -p)"
  log "arm $arm done"
}

log "=== PR33 in-model A/B start RECIPE=$RECIPE ARMS=[$ARMS] PCI-SERR=$(sel_count) uptime=$(uptime -p) ==="
hygiene || { log "initial hygiene failed"; exit 1; }
for arm in $ARMS; do
  run_arm "$arm" || log "ARM FAIL: $arm"
  [ -f "$RUN/IM_STOP" ] && { log "stopping: PCI SERR"; break; }
done
log "=== PR33 in-model A/B done PCI-SERR=$(sel_count) ==="
echo DONE >"$RUN/im_ab.status"
