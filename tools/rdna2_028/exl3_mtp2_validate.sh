#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# EXL3 27B mul1 + MTP=2 validation driver: boot (eager or F&P), coherence,
# the four production cells, and SpecDecoding acceptance, with a PCI-SERR
# gate. One engine at a time, own per-run cache root.
#
# Usage: EAGER=1 TAG=... PORT=... bash exl3_mtp2_validate.sh
set -uo pipefail

V=${VENV:-/home/chenco_adm/Apps/vllm/venv-7.14.0_0.28.0}
T=${VLLM_TREE:-/home/chenco_adm/vllm-rdna-0.28.0}
OUT=${OUT:-/home/chenco_adm/w4a8_runs}
TAG=${TAG:-2026-10-01_exl3-27b-mtp2}
PORT=${PORT:-18107}
MTP=${MTP:-2}
EAGER=${EAGER:-0}
CG_MODE=${CG_MODE:-FULL_AND_PIECEWISE}
HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-4,5,6,7}
MODEL=${MODEL:-/home/chenco_adm/models/Qwen3.8-27B-exl3-3.00bpw}
TRITON_CACHE=${TRITON_CACHE:-$T/cache/triton}
CELLS=${CELLS:-"1 1024 512 301|8 1024 512 302|1 16384 1024 303|8 16384 1024 304"}
D=$OUT/$TAG
CACHE=$D/cache
mkdir -p "$D/cells"

ROCM_SDK_LIB=$V/lib/python3.12/site-packages/_rocm_sdk_libraries/lib
ROCM_SDK=$V/lib/python3.12/site-packages/_rocm_sdk_core/lib
export LD_LIBRARY_PATH="$ROCM_SDK_LIB:$ROCM_SDK/host-math/lib:$ROCM_SDK/rocm_sysdeps/lib:$ROCM_SDK/core/lib:$V/lib/python3.12/site-packages/torch/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$D/driver.log"; }
sel_count() { sudo -n ipmitool sel list 2>/dev/null | grep -c 'PCI SERR'; }

our_pids() {
  local p fd
  for p in $(pgrep -f "entrypoints.openai.api_server.*exl3-27b-mul1" 2>/dev/null); do echo "$p"; done
  for p in $(pgrep -f "VLLM::EngineCore|VLLM::Worker_TP" 2>/dev/null); do
    fd=$(readlink "/proc/$p/fd/1" 2>/dev/null || true)
    case "$fd" in "$D"/serve.log) echo "$p" ;; esac
  done
}
our_pgids() { local p; for p in $(our_pids); do ps -o pgid= -p "$p" 2>/dev/null | tr -d ' '; done | sort -u; }
cleanup() {
  local pg
  for pg in $(our_pgids); do kill -TERM "-$pg" 2>/dev/null; done
  sleep 6
  for pg in $(our_pgids); do kill -KILL "-$pg" 2>/dev/null; done
  return 0
}
trap cleanup EXIT

log "=== EXL3 MTP=$MTP validate start (EAGER=$EAGER CG_MODE=$CG_MODE) tag=$TAG host=$(hostname) uptime=$(uptime -p) ==="
SERR0=$(sel_count); echo "$SERR0" >"$D/serr_baseline.txt"; log "PCI-SERR baseline=$SERR0"
cleanup
if ss -ltn 2>/dev/null | grep -q ":$PORT "; then log "port $PORT bound"; exit 1; fi

setsid nohup env \
  VENV="$V" VLLM_TREE="$T" MODEL="$MODEL" MTP="$MTP" PORT="$PORT" EAGER="$EAGER" CG_MODE="$CG_MODE" \
  HIP_VISIBLE_DEVICES="$HIP_VISIBLE_DEVICES" \
  VLLM_CAUSAL_CONV1D_RDNA2_FWD=0 VLLM_CAUSAL_CONV1D_RDNA2_UPDATE=0 \
  VLLM_CACHE_ROOT="$CACHE/vllm" TORCHINDUCTOR_CACHE_DIR="$CACHE/inductor" \
  TRITON_CACHE_DIR="$TRITON_CACHE" TORCH_EXTENSIONS_DIR="$T/cache/extensions" \
  bash "$T/tools/rdna/serve_gfx1030_exl3_27b.sh" >"$D/serve.log" 2>&1 </dev/null &

ready=0; dead=0
for i in $(seq 1 150); do
  sleep 10
  curl -s --max-time 4 "http://127.0.0.1:$PORT/v1/models" 2>/dev/null | grep -q exl3-27b-mul1 && { ready=1; break; }
  if [ -n "$(our_pids)" ]; then dead=0; else dead=$((dead + 1)); fi
  [ "$dead" -ge 4 ] && { log "server died (dead=$dead)"; break; }
done
if [ "$ready" != "1" ]; then
  log "NOT READY"
  grep -a -i "fc.suh\|no module or parameter\|Traceback\|ValueError\|Error" "$D/serve.log" | tail -20 | tee -a "$D/driver.log"
  exit 1
fi
log "READY"
grep -h "TunableOp lookup enabled" "$D/serve.log" | head -1 | tee -a "$D/driver.log"
grep -ah "SpecDecoding metrics" "$D/serve.log" | wc -l | sed 's/^/specdecode_lines_at_ready=/' | tee -a "$D/driver.log"

"$V/bin/python" - "$PORT" >"$D/coherence.txt" 2>&1 <<'PY'
import json, sys, urllib.request
port = sys.argv[1]
def chat(p, mt=64):
    b = json.dumps({"model": "exl3-27b-mul1", "messages": [{"role": "user", "content": p}],
                    "max_tokens": mt, "temperature": 0.0}).encode()
    r = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", data=b,
                               headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(r, timeout=600) as resp:
        return json.load(resp)["choices"][0]["message"]["content"] or ""
ok = True
for name, p, exp in [("france", "The capital of France is", "paris"), ("math", "2 + 2 =", "4")]:
    try: t = chat(p)
    except Exception as e: t = f"<ERROR {e!r}>"
    good = exp in t.lower(); ok = ok and good
    print(f"[{name}] coherent={good} | {t[:150]!r}")
print("COHERENCE_RESULT:", "PASS" if ok else "FAIL")
PY
log "coherence | $(grep -c 'coherent=True' "$D/coherence.txt")/2"

log "warmup c=1 in=1024 out=128"
( cd "$D" && "$V/bin/python" -m vllm.entrypoints.cli.main bench serve \
  --backend openai --endpoint /v1/completions --base-url "http://127.0.0.1:$PORT" \
  --model "$MODEL" --served-model-name exl3-27b-mul1 \
  --dataset-name random --random-input-len 1024 --random-output-len 128 \
  --num-prompts 1 --max-concurrency 1 --ignore-eos --request-rate inf \
  --seed 900 --temperature 0 ) >"$D/warmup.log" 2>&1

IFS='|' read -ra _cells <<<"$CELLS"
for cell in "${_cells[@]}"; do
  set -- $cell
  N=$1; IN=$2; OUTL=$3; SEED=$4
  CD="$D/cells/c${IN}_${N}"
  log "bench c=$N in=$IN"
  before=$(grep -ac "SpecDecoding metrics" "$D/serve.log" || true)
  ( cd "$D" && "$V/bin/python" -m vllm.entrypoints.cli.main bench serve \
    --backend openai --endpoint /v1/completions --base-url "http://127.0.0.1:$PORT" \
    --model "$MODEL" --served-model-name exl3-27b-mul1 \
    --dataset-name random --random-input-len "$IN" --random-output-len "$OUTL" \
    --num-prompts "$N" --max-concurrency "$N" --ignore-eos --request-rate inf \
    --seed "$SEED" --temperature 0 \
    --save-result --result-dir "$CD" ) >"$CD.log" 2>&1
  log "  c=$N in=$IN done: $(grep -m1 'Output token throughput' "$CD.log" | tr -s ' ')"
  sleep 12
  grep -a "SpecDecoding metrics" "$D/serve.log" | tail -n +$((before + 1)) >"$CD.acceptance.txt"
  tail -1 "$CD.acceptance.txt" | tee -a "$D/driver.log"
  c=$(sel_count); log "  SEL PCI-SERR=$c"
  if [ "$c" -gt "$SERR0" ]; then
    log "STOP new PCI SERR ($c > $SERR0)"
    echo STOP >"$D/STOP"
    break
  fi
done

grep -ah "SpecDecoding metrics" "$D/serve.log" >"$D/acceptance.txt"
log "=== EXL3 MTP=$MTP validate done ==="
echo "DONE" >"$D/status.txt"
