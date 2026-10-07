#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Generic in-model A/B driver over tools/rdna/serve_rdna.sh RECIPE=...
#
# One arm per invocation: boot the recipe, warm, coherence probe, run cells,
# tear down. Swap only the thing under test (VLLM_CAUSAL_CONV1D_RDNA2_*,
# ATTN, or the installed _rocm_C .so) between arms; keep the compile cache
# shared so the graph is otherwise byte-identical.
#
#   RECIPE=flashnext-mtp0 SERVED=flash-next ARM=triton CONV1D=0 \
#     CACHE=$T/cache/wt-ab-fn PORT=18150 TAG=item2_triton \
#     CELLS="1 16384 512 711|1 1024 512 721" bash tools/rdna2_028/wt_ab.sh
#
# Env knobs:
#   RECIPE SERVED MODEL PORT ARM TAG CACHE KEEP_CACHE CELLS WARMUP READY_MAX
#   SO           optional .so to install into vllm/_rocm_C.abi3.so
#   CONV1D       optional 0|1 -> both VLLM_CAUSAL_CONV1D_RDNA2_{FWD,UPDATE}
#   ATTN         optional fa|triton|none
#   EXTRA_ARGS   extra trailing KEY=value overrides for serve_rdna.sh
#   PLE          optional VLLM_PLE_QUANT_DIR (only set when non-empty)
#   HIP_VISIBLE  GPU list (default 4,5,6,7)
#
# No /tmp. Logs: /home/chenco_adm/w4a8_runs/<TAG>/.
set -uo pipefail

V=${V:-/home/chenco_adm/Apps/vllm/venv-7.14.0_0.28.0}
T=${T:-/home/chenco_adm/vllm-rdna-0.28.0}
RUN=${RUN:-/home/chenco_adm/w4a8_runs}
RECIPE=${RECIPE:?RECIPE required}
SERVED=${SERVED:-}
ARM=${ARM:-arm}
PORT=${PORT:-18150}
TAG=${TAG:-wt_${RECIPE}_${ARM}}
CELLS=${CELLS:-"1 16384 512 711|1 1024 512 721"}
WARMUP=${WARMUP:-"1024 64 991|16384 64 992"}
READY_MAX=${READY_MAX:-180}
KEEP_CACHE=${KEEP_CACHE:-0}
HIP_VISIBLE=${HIP_VISIBLE:-4,5,6,7}
CACHE=${CACHE:-$T/cache/wt-ab-$RECIPE}

D=$RUN/$TAG
mkdir -p "$D/cells"
DRIVER_LOG=$D/driver.log
log(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$DRIVER_LOG"; }
sel_count(){ timeout 10 sudo -n ipmitool sel list 2>/dev/null | grep -c 'PCI SERR'; }

our_pids() {
  local p e
  for p in $(pgrep -f "entrypoints.cli.main serve|entrypoints.openai.api_server|VLLM::Worker|VLLM::EngineCore|PleOffloadWorker" 2>/dev/null); do
    e=$(tr '\0' '\n' <"/proc/$p/environ" 2>/dev/null || true)
    case "$e" in *"VLLM_CACHE_ROOT=$CACHE"*) echo "$p" ;; esac
  done
}
hygiene() {
  local n=0 left p i
  for p in $(our_pids); do kill -TERM "$p" 2>/dev/null && n=$((n+1)); done
  [ "$n" -gt 0 ] && sleep 8
  for p in $(our_pids); do kill -KILL "$p" 2>/dev/null; done
  for i in $(seq 1 30); do left=$(our_pids | tr '\n' ' '); [ -z "${left// /}" ] && break; sleep 2; done
  left=$(our_pids | tr '\n' ' ')
  log "hygiene: terminated=$n remaining=[${left% }]"
  [ -z "${left// /}" ]
}

if [[ -n ${SO:-} ]]; then
  [ -r "$SO" ] || { log "missing SO=$SO"; exit 1; }
  cp "$SO" "$T/vllm/_rocm_C.abi3.so"
fi
so=$(sha256sum "$T/vllm/_rocm_C.abi3.so" | cut -c1-16)
log "=== ARM=$ARM recipe=$RECIPE served=$SERVED port=$PORT so=$so conv1d=${CONV1D:-} attn=${ATTN:-} ==="
hygiene || { log "hygiene failed"; exit 1; }
if ss -ltn 2>/dev/null | grep -q ":$PORT "; then log "port $PORT busy"; exit 1; fi

if [ "$KEEP_CACHE" = 1 ] && [ -d "$CACHE/inductor" ]; then
  log "KEEP_CACHE=1: reusing $CACHE"
else
  rm -rf "$CACHE"
fi
mkdir -p "$CACHE/inductor" "$CACHE/extensions" "$CACHE/triton" "$CACHE/vllm"
rm -f "$CACHE/rdna_ar_wedged"

# Trailing KEY=value overrides passed through to serve_rdna.sh.
declare -a ov=()
[[ -n ${SERVED:-} ]] && ov+=(SERVED_NAME="$SERVED")
[[ -n ${CONV1D:-} ]] && ov+=(VLLM_CAUSAL_CONV1D_RDNA2_FWD="$CONV1D" VLLM_CAUSAL_CONV1D_RDNA2_UPDATE="$CONV1D")
[[ -n ${ATTN:-} ]] && ov+=(ATTN="$ATTN")
# shellcheck disable=SC2206
[[ -n ${EXTRA_ARGS:-} ]] && ov+=($EXTRA_ARGS)

S0=$(sel_count); log "PCI-SERR before=$S0 uptime=$(uptime -p)"
T0=$(date +%s)
setsid nohup env \
  VENV="$V" VLLM_TREE="$T" MODEL="${MODEL:-}" ${PLE:+VLLM_PLE_QUANT_DIR="$PLE"} \
  VLLM_CACHE_ROOT="$CACHE" TORCHINDUCTOR_CACHE_DIR="$CACHE/inductor" \
  TRITON_CACHE_DIR="$CACHE/triton" TORCH_EXTENSIONS_DIR="$CACHE/extensions" \
  HIP_VISIBLE_DEVICES="$HIP_VISIBLE" \
  bash "$T/tools/rdna/serve_rdna.sh" RECIPE="$RECIPE" PORT="$PORT" "${ov[@]}" \
  >"$D/serve.log" 2>&1 </dev/null &
ready=0; dead=0; health=""
for i in $(seq 1 "$READY_MAX"); do
  sleep 10
  models=$(curl -s --max-time 4 "http://127.0.0.1:$PORT/v1/models" 2>/dev/null)
  if [ -n "$models" ]; then
    if [ -n "$SERVED" ]; then echo "$models" | grep -q "$SERVED" && { ready=1; break; }
    else ready=1; break; fi
  fi
  if [ -n "$(our_pids)" ]; then dead=0; else dead=$((dead+1)); fi
  [ "$dead" -ge 3 ] && { log "server gone during boot"; break; }
done
COLD=$(( $(date +%s) - T0 ))
if [ "$ready" != 1 ]; then log "NOT READY after ${COLD}s"; tail -50 "$D/serve.log" >>"$DRIVER_LOG"; hygiene; exit 1; fi
log "READY t=${COLD}s"; echo "$COLD" >"$D/cold_seconds.txt"
S1=$(sel_count); log "PCI-SERR after boot=$S1 uptime=$(uptime -p)"
if [ "$S1" -gt "$S0" ]; then echo "STOP NEW PCI SERR" >"$RUN/AB_STOP"; hygiene; exit 2; fi

# attention backend actually in use
grep -iE "attention backend|pinning the attention backend|Using .* backend" "$D/serve.log" | tail -3 >>"$DRIVER_LOG" || true

if [ "${BOOT_ONLY:-0}" = 1 ]; then log "BOOT_ONLY ready"; exit 0; fi

if [ -z "$SERVED" ]; then
  SERVED=$(grep -m1 -oE '"id":"[^"]+"' "$D/serve.log" 2>/dev/null | head -1 | cut -d'"' -f4 || true)
fi

for wl in $WARMUP; do set -- $wl
  ( cd "$D" && "$V/bin/python" -m vllm.entrypoints.cli.main bench serve --backend openai --endpoint /v1/completions --base-url "http://127.0.0.1:$PORT" --model "${MODEL:-$SERVED}" --served-model-name "$SERVED" --dataset-name random --random-input-len "$1" --random-output-len "$2" --num-prompts 1 --max-concurrency 1 --ignore-eos --request-rate inf --seed "$3" --temperature 0 ) >"$D/warmup_$1.log" 2>&1
done
log "warmup done"

"$V/bin/python" - "$PORT" "$SERVED" >"$D/coherence.txt" 2>&1 <<'PY'
import json,sys,urllib.request,re
port,model=sys.argv[1],sys.argv[2]
def ask(p,mt=24):
    body=json.dumps({"model":model,"prompt":p,"max_tokens":mt,"temperature":0.0,"ignore_eos":True}).encode()
    r=urllib.request.Request(f"http://127.0.0.1:{port}/v1/completions",data=body,headers={"Content-Type":"application/json"})
    with urllib.request.urlopen(r,timeout=900) as resp: return json.load(resp)["choices"][0]["text"] or ""
def garbage(t):
    if not t.strip(): return "empty"
    if re.search(r'(.)\1{29,}', t): return "repeat>=30"
    if sum(1 for c in t if not c.isprintable() and c not in "\n\t") > len(t)*0.1: return "nonprintable"
    if t.count("!")>len(t)*0.5: return "bang-flood"
    return "ok"
ok=0; tot=0
for name,p,exp in [("france","The capital of France is","paris"),("math","2 + 2 =","4")]:
    tot+=1
    try: t=ask(p)
    except Exception as e: t=f"<ERR {e!r}>"
    g=garbage(t); good=(exp in t.lower()) and g=="ok"; ok+=good
    print(f"[{name}] good={good} garbage={g} exp={exp!r} | {t[:100]!r}")
print(f"COHERENCE {ok}/{tot}")
PY
log "coherence: $(grep -m1 COHERENCE "$D/coherence.txt" 2>/dev/null)"

IFS='|' read -ra _cells <<<"$CELLS"
for cell in "${_cells[@]}"; do set -- $cell
  N=$1; IN=$2; OUTL=$3; SEED=$4
  CD="$D/cells/c${N}_${IN}_s${SEED}"
  mkdir -p "$CD"
  ( cd "$D" && "$V/bin/python" -m vllm.entrypoints.cli.main bench serve --backend openai --endpoint /v1/completions --base-url "http://127.0.0.1:$PORT" --model "${MODEL:-$SERVED}" --served-model-name "$SERVED" --dataset-name random --random-input-len "$IN" --random-output-len "$OUTL" --num-prompts "$N" --max-concurrency "$N" --ignore-eos --request-rate inf --seed "$SEED" --temperature 0 --save-result --result-dir "$CD" ) >"$CD.log" 2>&1
  log "cell c=$N in=$IN s=$SEED | $(grep -m1 'Output token throughput' "$CD.log" | tr -s ' ') | $(grep -m1 'Mean TPOT' "$CD.log" | tr -s ' ')"
  S2=$(sel_count); if [ "$S2" -gt "$S0" ]; then echo "STOP NEW PCI SERR" >"$RUN/AB_STOP"; break; fi
done

{ echo "cell,arm,seed,requests,output_tok_s,total_tok_s,mean_ttft_ms,mean_tpot_ms,median_itl_ms,p99_itl_ms"
  for cell in "${_cells[@]}"; do set -- $cell
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
print(f"c{n}_{inn},{arm},{seed},{g('completed','successful_requests')},{g('output_throughput')},{g('total_token_throughput')},{g('mean_ttft_ms','mean_ttft')},{g('mean_tpot_ms','mean_tpot')},{g('median_itl_ms')},{g('p99_itl_ms')}")
PY
  done; } >"$D/summary.csv" 2>"$D/summary.err"

hygiene
S3=$(sel_count); log "PCI-SERR after teardown=$S3 uptime=$(uptime -p)"
log "arm $ARM done"
