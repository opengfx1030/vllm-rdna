#!/usr/bin/env bash
# PR33 in-model A/B driver v2 (takeover): one arm per invocation.
# Adds: warm-cache copy (identical compile cache across arms, only the .so
# differs), a longer readiness timeout, per-arm result dirs.
#
#   ARM=baseline|patched RECIPE=flashnext-mtp0 [WARM_FROM=<arm>] PORT=18121 \
#     bash im_ab2.sh
#
# No /tmp. Stop on new PCI SERR.
set -uo pipefail
V=${V:-/home/chenco_adm/Apps/vllm/venv-7.14.0_0.28.0}
T=${T:-/home/chenco_adm/vllm-rdna-0.28.0}
RUN=${RUN:-/home/chenco_adm/w4a8_runs/pr33_validate}
MODEL=${MODEL:-/home/chenco_adm/hfcache/hub/models--wtdcode--Qwen3.8-Flash-Next-AWQ-W4A16/snapshots/0939125b929543a783ce700c90e36dd1a575c00c}
PLE=${PLE:-/home/chenco_adm/hfcache/hub/models--primitive-ai--Qwen3.8-Flash-Next-PLE-quant/snapshots/4f861b63f69e61bfc2e22130ec91ec67f03ec43e/ples_int4}
ARM=${ARM:?ARM required}
RECIPE=${RECIPE:-flashnext-mtp0}
PORT=${PORT:-18121}
WARM_FROM=${WARM_FROM:-}
READY_MAX=${READY_MAX:-180}          # x 10s
CELLS=${CELLS:-"1 1024 512 701|2 1024 512 702|3 1024 512 703|1 16384 1024 711|2 16384 1024 712|3 16384 1024 713|8 1024 512 708|8 16384 1024 718"}
TAG=${TAG:-im2_$ARM}
D=$RUN/$TAG
DRIVER_LOG=$D/driver.log
mkdir -p "$D/cells"
log(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$DRIVER_LOG"; }

# Identify processes owned by THIS A/B campaign. /proc/PID/exe resolves to the
# system python (venv symlink) and worker cmdlines are renamed to
# "VLLM::Worker_*", so neither is reliable; match the campaign cache prefix in
# the environment instead. The prefix covers every arm (so a prior arm's engine
# is reaped before the next boots) while staying co-tenant-safe.
our_pids() {
  local p e
  for p in $(pgrep -f "entrypoints.cli.main serve|entrypoints.openai.api_server|VLLM::Worker|VLLM::EngineCore|PleOffloadWorker" 2>/dev/null); do
    e=$(tr '\0' '\n' <"/proc/$p/environ" 2>/dev/null || true)
    case "$e" in *"VLLM_CACHE_ROOT=$T/cache/ab-pr33-"*) echo "$p" ;; esac
  done
}
hygiene() {
  local n=0 left p i
  for p in $(our_pids); do kill -TERM "$p" 2>/dev/null && n=$((n+1)); done
  [ "$n" -gt 0 ] && sleep 8
  for p in $(our_pids); do kill -KILL "$p" 2>/dev/null; done
  for i in $(seq 1 30); do
    left=$(our_pids | tr '\n' ' ')
    [ -z "${left// /}" ] && break
    sleep 2
  done
  left=$(our_pids | tr '\n' ' ')
  log "hygiene: terminated=$n remaining=[${left% }]"
  [ -z "${left// /}" ]
}
sel_count(){ timeout 10 sudo -n ipmitool sel list 2>/dev/null | grep -c 'PCI SERR'; }

CACHE=${CACHE:-$T/cache/${CACHE_NAME:-ab-pr33-$ARM}}
SO=$RUN/_rocm_C.$ARM.so
[ -r "$SO" ] || { log "missing $SO"; exit 1; }
cp "$SO" "$T/vllm/_rocm_C.abi3.so"
so=$(sha256sum "$T/vllm/_rocm_C.abi3.so" | cut -c1-16)
log "=== ARM=$ARM so=$so recipe=$RECIPE port=$PORT warm=${WARM_DIR:-none} ==="
hygiene || { log "hygiene failed"; exit 1; }
if ss -ltn 2>/dev/null | grep -q ":$PORT "; then log "port $PORT busy"; exit 1; fi

if [ "${KEEP_CACHE:-0}" = 1 ]; then
  log "KEEP_CACHE=1: reusing $CACHE"
else
  rm -rf "$CACHE"
fi
mkdir -p "$CACHE"
if [ -n "${WARM_DIR:-}" ] && [ -d "${WARM_DIR:-}" ]; then
  cp -a "$WARM_DIR/." "$CACHE/"
  log "warm cache copied from $WARM_DIR"
fi
mkdir -p "$CACHE/inductor" "$CACHE/extensions" "$CACHE/triton" "$CACHE/vllm"
rm -f "$CACHE/rdna_ar_wedged"

S0=$(sel_count); log "PCI-SERR before=$S0 uptime=$(uptime -p)"
T0=$(date +%s)
setsid nohup env \
  VENV="$V" VLLM_TREE="$T" MODEL="$MODEL" VLLM_PLE_QUANT_DIR="$PLE" \
  VLLM_CACHE_ROOT="$CACHE" TORCHINDUCTOR_CACHE_DIR="$CACHE/inductor" \
  TRITON_CACHE_DIR="$CACHE/triton" TORCH_EXTENSIONS_DIR="$CACHE/extensions" \
  HIP_VISIBLE_DEVICES=4,5,6,7 \
  bash "$T/scripts/serve_rdna.sh" RECIPE="$RECIPE" PORT="$PORT" \
  >"$D/serve.log" 2>&1 </dev/null &
ready=0; dead=0; i=0
for i in $(seq 1 "$READY_MAX"); do
  sleep 10
  curl -s --max-time 4 "http://127.0.0.1:$PORT/v1/models" 2>/dev/null | grep -q flash-next && { ready=1; break; }
  if [ -n "$(our_pids)" ]; then dead=0; else dead=$((dead+1)); fi
  [ "$dead" -ge 3 ] && { log "server gone during boot"; break; }
done
COLD=$(( $(date +%s) - T0 ))
if [ "$ready" != 1 ]; then log "NOT READY after ${COLD}s"; tail -30 "$D/serve.log" >>"$DRIVER_LOG"; hygiene; exit 1; fi
log "READY t=${COLD}s"; echo "$COLD" >"$D/cold_seconds.txt"
S1=$(sel_count); log "PCI-SERR after boot=$S1 uptime=$(uptime -p)"
if [ "$S1" -gt "$S0" ]; then echo "STOP NEW PCI SERR" >"$RUN/IM_STOP"; hygiene; exit 2; fi
if [ "${BOOT_ONLY:-0}" = 1 ]; then log "BOOT_ONLY: server ready on $PORT"; exit 0; fi

# Warmup (throwaway)
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
  S2=$(sel_count); if [ "$S2" -gt "$S0" ]; then echo "STOP NEW PCI SERR" >"$RUN/IM_STOP"; break; fi
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

hygiene
S3=$(sel_count); log "PCI-SERR after teardown=$S3 uptime=$(uptime -p)"
log "arm $arm done"
