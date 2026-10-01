#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# ONE arm of the Qwen3.8-Flash-Next W4A8/W4A16 matrix on gfx1030.
#
# Launcher: scripts/serve_gfx1030_flashnext_mtp.sh (the documented Flash-Next
# MTP path: V2 runner, FA-RDNA2, MTP ladder [3,6,12,24] + local argmax, RDNA AR
# one-shot <=64 KiB, prefix caching, FULL_AND_PIECEWISE, TP=4/EP4, PLE offload).
# This driver adds the one thing the launcher does not own: an arm-tagged,
# isolated graph/Triton cache + the W4A8 opt-in env + measured cells.
#
# Env:
#   ARM=w4a8on|w4a8off   cache suffix (arm dir = $T/cache/arm-fn-$ARM)
#   W4A8=1|0             VLLM_RDNA2_W4A8_SDOT4
#   MTP=0|2              speculative tokens (launcher builds the capture ladder)
#   TAG=...              output dir under $OUT
#   PORT, SERR_BASE, OUT, MODEL, CELLS, WARM_TRITON, DO_WARMUP
#
# Guards: hygiene scoped to the 0.28.0 venv only (never a co-tenant on
# venv-7.14.0), IPMI PCI-SERR baseline before/after every phase, stop marker.
# No /tmp: everything under /home/chenco_adm/w4a8_runs/<TAG>/.
set -uo pipefail

V=/home/chenco_adm/Apps/vllm/venv-7.14.0_0.28.0
T=/home/chenco_adm/vllm-rdna-0.28.0
MODEL=${MODEL:-/home/chenco_adm/hfcache/hub/models--wtdcode--Qwen3.8-Flash-Next-AWQ-W4A16/snapshots/0939125b929543a783ce700c90e36dd1a575c00c}
ARM=${ARM:?set ARM (w4a8on|w4a8off)}
W4A8=${W4A8:?set W4A8 (1|0)}
MTP=${MTP:-0}
TAG=${TAG:-2026-09-29_fn-$ARM-m$MTP}
PORT=${PORT:-18260}
OUT=${OUT:-/home/chenco_adm/w4a8_runs}
CACHE=$T/cache/arm-fn-$ARM
TRITON_TEMPLATE=${TRITON_TEMPLATE:-$T/cache/triton}
WARM_TRITON=${WARM_TRITON:-1}
DO_WARMUP=${DO_WARMUP:-1}
CELLS=${CELLS:-"1 1024 512 221|8 1024 512 222|1 16384 1024 111|8 16384 1024 112"}
D=$OUT/$TAG
mkdir -p "$D/cells"

ROCM_SDK_LIB=$V/lib/python3.12/site-packages/_rocm_sdk_libraries/lib
ROCM_SDK=$V/lib/python3.12/site-packages/_rocm_sdk_core/lib
export LD_LIBRARY_PATH="$ROCM_SDK_LIB:$ROCM_SDK/host-math/lib:$ROCM_SDK/rocm_sysdeps/lib:$ROCM_SDK/core/lib:$V/lib/python3.12/site-packages/torch/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$D/driver.log"; }

# --- OUR processes only: the 0.28.0 venv, never the co-tenant's venv-7.14.0 ---
our_pids() {
  local p exe
  for p in $(pgrep -f "entrypoints.cli.main serve|entrypoints.openai.api_server|VLLM::Worker|VLLM::EngineCore|PleOffloadWorker" 2>/dev/null); do
    exe=$(readlink -f "/proc/$p/exe" 2>/dev/null || true)
    case "$exe" in
      */venv-7.14.0_0.28.0/bin/python*) echo "$p" ;;
    esac
  done
}

show_pids() {
  local p exe cwd
  for p in $(our_pids); do
    exe=$(readlink -f "/proc/$p/exe" 2>/dev/null)
    cwd=$(readlink -f "/proc/$p/cwd" 2>/dev/null)
    echo "  pid=$p exe=$exe cwd=$cwd cmd=$(tr '\0' ' ' </proc/$p/cmdline 2>/dev/null | cut -c1-70)"
  done
}

hygiene() {
  local n=0 p left
  show_pids | sed 's/^/hyg:before /' | tee -a "$D/driver.log"
  for p in $(our_pids); do kill -TERM "$p" 2>/dev/null && n=$((n + 1)); done
  [ "$n" -gt 0 ] && sleep 8
  for p in $(our_pids); do kill -KILL "$p" 2>/dev/null; done
  sleep 2
  left=$(our_pids | tr '\n' ' ')
  log "hygiene: terminated=$n remaining_our=[${left% }]"
  {
    echo "=== [$(date -Is)] KFD map ==="
    sudo -n rocm-smi --showpids 2>/dev/null | grep -E "^\s+[0-9]+" || true
  } >>"$D/kfd.log"
  [ -n "${left// /}" ] && { log "HYGIENE FAIL: $left"; return 1; }
  return 0
}

sel_count() { sudo -n ipmitool sel list 2>/dev/null | grep -c 'PCI SERR'; }
sel_snap() {
  local tag=$1 c
  c=$(sel_count)
  log "SEL[$tag] PCI-SERR=$c base=${SERR0:-?} uptime=$(uptime -p)"
  if [ -n "${SERR0:-}" ] && [ "${c:-0}" -gt "$SERR0" ]; then
    echo "STOP NEW PCI SERR after $tag ($c > $SERR0)" >"$D/STOP"
    log "STOP: new PCI SERR after $tag"
    return 1
  fi
  return 0
}

teardown() {
  local api p
  api=$(grep -oE "APIServer pid=[0-9]+" "$D/serve.log" 2>/dev/null | head -1 | cut -d= -f2)
  [ -n "$api" ] && kill -TERM "$api" 2>/dev/null && sleep 10
  for p in $(our_pids); do kill -TERM "$p" 2>/dev/null; done
  sleep 6
  for p in $(our_pids); do kill -KILL "$p" 2>/dev/null; done
  sleep 2
}

# --- Preflight -------------------------------------------------------------
log "=== ARM=$ARM W4A8=$W4A8 MTP=$MTP TAG=$TAG PORT=$PORT RDNA_AR=1 (one-shot 64K) ==="
log "preflight uptime=$(uptime -p) host=$(hostname)"
hygiene || { log "aborting: hygiene failed"; exit 1; }
if ss -ltn 2>/dev/null | grep -q ":$PORT "; then log "aborting: port $PORT bound"; exit 1; fi
rm -rf "$CACHE"
mkdir -p "$CACHE/inductor" "$CACHE/extensions"
if [ "$WARM_TRITON" = "1" ] && [ -d "$TRITON_TEMPLATE" ]; then
  cp -a "$TRITON_TEMPLATE" "$CACHE/triton"
  log "triton cache seeded from $TRITON_TEMPLATE ($(ls "$CACHE/triton" | wc -l) entries)"
else
  mkdir -p "$CACHE/triton"; log "triton cache FRESH (cold autotune will run)"
fi
rm -f "$CACHE/rdna_ar_wedged"
[ -n "${SERR_BASE:-}" ] && SERR0=$SERR_BASE || SERR0=$(sel_count)
log "arm cache=$CACHE baseline PCI-SERR=$SERR0"
sel_snap preflight || exit 1

# --- Launch ----------------------------------------------------------------
T0=$(date +%s)
setsid nohup env \
  VENV="$V" VLLM_TREE="$T" MTP="$MTP" ATTN=fa TP=4 PORT="$PORT" MODEL="$MODEL" \
  VLLM_CACHE_ROOT="$CACHE" TORCHINDUCTOR_CACHE_DIR="$CACHE/inductor" \
  TRITON_CACHE_DIR="$CACHE/triton" TORCH_EXTENSIONS_DIR="$CACHE/extensions" \
  VLLM_RDNA2_W4A8_SDOT4="$W4A8" \
  VLLM_RDNA_AR=1 VLLM_RDNA_AR_MAX_KB=64 VLLM_RDNA_AR_ONESHOT_KB=64 \
  VLLM_RDNA_AR_BLOCKS=0 VLLM_RDNA_AR_PACE=0 VLLM_FORCE_CUSTOM_ALL_REDUCE=0 \
  NCCL_P2P_LEVEL=pxb RCCL_P2P_NET_DISABLE=1 RCCL_P2P_BATCH_ENABLE=1 \
  NCCL_PROTO=Simple RCCL_MSCCL_ENABLE=0 PYTHONFAULTHANDLER=1 \
  HIP_VISIBLE_DEVICES=0,1,2,3 \
  bash "$T/scripts/serve_gfx1030_flashnext_mtp.sh" >"$D/serve.log" 2>&1 </dev/null &
log "flashnext launched cache=$CACHE log=$D/serve.log"

ready=0
dead=0
for i in $(seq 1 90); do
  sleep 10
  curl -s --max-time 4 "http://127.0.0.1:$PORT/v1/models" 2>/dev/null | grep -q flash-next && { ready=1; break; }
  [ -f "$D/STOP" ] && break
  if pgrep -f "[a]pi_server" >/dev/null; then dead=0; else dead=$((dead + 1)); fi
  if [ "$dead" -ge 3 ]; then log "server process gone during boot (after $((i*10))s)"; break; fi
done
COLD=$(( $(date +%s) - T0 ))
if [ "$ready" != "1" ]; then
  log "NOT READY after ${COLD}s"
  tail -40 "$D/serve.log" | tee -a "$D/driver.log"
  sel_snap notready
  teardown; hygiene
  echo "NOT_READY" >"$D/status.txt"
  exit 1
fi
log "READY t=${COLD}s"
echo "$COLD" >"$D/cold_compile_seconds.txt"
sel_snap after_boot || { teardown; hygiene; exit 1; }

# --- Warmup (throwaway: cold-start pollution is real on this stack) ---------
if [ "$DO_WARMUP" = "1" ]; then
  for wl in "1024 64 999" "16384 64 998"; do
    set -- $wl
    log "warmup in=$1 out=$2"
    ( cd "$D" && "$V/bin/python" -m vllm.entrypoints.cli.main bench serve \
      --backend openai --endpoint /v1/completions --base-url "http://127.0.0.1:$PORT" \
      --model "$MODEL" --served-model-name flash-next --dataset-name random \
      --random-input-len "$1" --random-output-len "$2" --num-prompts 1 --max-concurrency 1 \
      --ignore-eos --request-rate inf --seed "$3" --temperature 0 \
      --save-result --result-dir "$D/warmup_$1" ) >"$D/warmup_$1.log" 2>&1
  done
  log "warmup done"
  sel_snap after_warmup || { teardown; hygiene; exit 1; }
fi

# --- Coherence (documented probe) ------------------------------------------
log "coherence probe start"
"$V/bin/python" "$T/tools/rdna2_028/probe_w4a8.py" "http://127.0.0.1:$PORT/v1" flash-next 1 >"$D/coherence.txt" 2>&1
log "coherence probe done rc=$?"
grep -E "^\[|OK /" "$D/coherence.txt" | tee -a "$D/driver.log"

# --- Measured cells --------------------------------------------------------
IFS='|' read -ra _cells <<<"$CELLS"
for cell in "${_cells[@]}"; do
  # CELLS field order matches the established protocol: "n_prompts in_len out_len seed".
  set -- $cell
  N=$1; IN=$2; OUTL=$3; SEED=$4
  if [ "$N" -gt 8 ] || [ "$IN" -lt 64 ] || [ "$OUTL" -lt 1 ]; then
    log "ABORT bad cell spec: N=$N IN=$IN OUT=$OUTL (want N<=8, IN>=64, OUT>=1)"
    echo "BAD_CELL_SPEC" >"$D/status.txt"
    teardown; hygiene
    exit 2
  fi
  CD="$D/cells/c${IN}_${N}"
  log "bench c=$N in=$IN out=$OUTL start"
  ( cd "$D" && "$V/bin/python" -m vllm.entrypoints.cli.main bench serve \
    --backend openai --endpoint /v1/completions --base-url "http://127.0.0.1:$PORT" \
    --model "$MODEL" --served-model-name flash-next --dataset-name random \
    --random-input-len "$IN" --random-output-len "$OUTL" --num-prompts "$N" --max-concurrency "$N" \
    --ignore-eos --request-rate inf --seed "$SEED" --temperature 0 \
    --save-result --result-dir "$CD" ) >"$CD.log" 2>&1
  log "bench c=$N in=$IN done: $(grep -m1 'Output token throughput' "$CD.log" | tr -s ' ')"
  # Per-cell coherence (France -> Paris, 2+2 -> 4) + garbage guard text captured raw.
  "$V/bin/python" - "$PORT" "$CD" <<'PY' >"$CD/coherence.txt" 2>&1
import json, sys, urllib.request
port, out = sys.argv[1], sys.argv[2]
def ask(prompt, mt=16):
    body = json.dumps({"model": "flash-next", "prompt": prompt, "max_tokens": mt,
                       "temperature": 0.0, "ignore_eos": True}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.load(r)["choices"][0]["text"] or ""
for name, prompt, expect in [("france", "The capital of France is", "paris"),
                             ("math", "2 + 2 =", "4")]:
    try:
        t = ask(prompt)
    except Exception as e:  # noqa: BLE001
        t = f"<ERROR {e!r}>"
    ok = expect in t.lower()
    print(f"[{name}] coherent={ok} | {t[:120]!r}")
PY
  sel_snap "after_c${N}_${IN}" || { teardown; hygiene; exit 1; }
done

# --- Markers ---------------------------------------------------------------
{
  echo "=== 'W4A8 sdot4 path active' (expected iff W4A8=1 AND a dense quantized linear ran) ==="
  grep -m4 "W4A8 sdot4 path active" "$D/serve.log" || echo "(absent)"
  echo "=== 'W4A8-DEBUG' fast-path shapes ==="
  grep -h "W4A8-DEBUG" "$D/serve.log" | sort -u | head -30 || echo "(none)"
  echo "=== rdna_ar lines (expected: one-shot active on every rank) ==="
  grep -h "rdna_ar:" "$D/serve.log" | sort -u || echo "(none)"
  echo "=== all-reduce backend selection ==="
  grep -m2 "all-reduce backends" "$D/serve.log" || echo "(none)"
  echo "=== attention backend ==="
  grep -m2 "attention backend\|Using RDNA_ATTN\|Attention backend" "$D/serve.log" || echo "(none)"
  echo "=== shm_broadcast stalls ==="
  grep -c "no available shared memory block" "$D/serve.log" || true
} >"$D/markers.txt" 2>&1
sed -n '1,40p' "$D/markers.txt" | tee -a "$D/driver.log"

{
  echo "=== SpecDecoding metrics (serve log) ==="
  grep "SpecDecoding metrics" "$D/serve.log" || echo "(none)"
} >"$D/acceptance.txt" 2>&1

# --- Extract (CSV) ---------------------------------------------------------
"$V/bin/python" "$T/tools/rdna2_028/flashnext_w4a8_extract.py" "$D" | tee "$D/cells.csv" | tail -8

teardown
hygiene
sel_snap after_teardown || true
log "done $TAG (W4A8=$W4A8 MTP=$MTP)"
echo "OK" >"$D/status.txt"
