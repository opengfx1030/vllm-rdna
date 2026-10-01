#!/usr/bin/env bash
# fp32-accum matrix driver — Flash-Next, TP=4, FULL_AND_PIECEWISE cudagraphs,
# prefix caching, FA-RDNA2, frozen TunableOp rows. Runs on GPUs 4-7.
#
# Arms (per-arm cells):
#   fp32-m0 / fp32-m2 : VLLM_RDNA2_MOE_FP32_ACCUM=1 (default), W4A8=0, 4 cells
#   cas-m0  / cas-m2  : VLLM_RDNA2_MOE_FP32_ACCUM=0 (legacy CAS), W4A8=0, c=1 cells
#   w4a8-m0           : VLLM_RDNA2_MOE_FP32_ACCUM=1 + VLLM_RDNA2_W4A8_SDOT4=1, 1 cell
#
# Env: ARMS (space list), OUT, MODEL, TAG_PREFIX, HIP_DEVICES. No /tmp.
# Logs + captures live under /home/chenco_adm/w4a8_runs/.
set -uo pipefail

V=/home/chenco_adm/Apps/vllm/venv-7.14.0_0.28.0
T=/home/chenco_adm/vllm-rdna-0.28.0
MODEL=${MODEL:-/home/chenco_adm/hfcache/hub/models--wtdcode--Qwen3.8-Flash-Next-AWQ-W4A16/snapshots/0939125b929543a783ce700c90e36dd1a575c00c}
OUT=${OUT:-/home/chenco_adm/w4a8_runs}
TAG_PREFIX=${TAG_PREFIX:-2026-09-30_fp32-accum-matrix}
HIP_DEVICES=${HIP_DEVICES:-4,5,6,7}
ARMS=${ARMS:-"fp32-m0 fp32-m2 cas-m0 cas-m2 w4a8-m0"}
TRITON_TEMPLATE=${TRITON_TEMPLATE:-$T/cache/triton}

ROCM_SDK_LIB=$V/lib/python3.12/site-packages/_rocm_sdk_libraries/lib
ROCM_SDK=$V/lib/python3.12/site-packages/_rocm_sdk_core/lib
export LD_LIBRARY_PATH="$ROCM_SDK_LIB:$ROCM_SDK/host-math/lib:$ROCM_SDK/rocm_sysdeps/lib:$ROCM_SDK/core/lib:$V/lib/python3.12/site-packages/torch/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

DRIVER_LOG=$OUT/fp32_accum_matrix.log
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$DRIVER_LOG"; }

# The api_server -> EngineCore spawn drops launcher-passed env vars, so the
# per-arm fork knobs (VLLM_RDNA2_MOE_FP32_ACCUM / VLLM_RDNA2_W4A8_SDOT4) never
# reach the workers via the launch env. This installs (idempotently) a venv
# sitecustomize hook that re-loads the per-arm mode file into `os.environ` in
# every process, which the worker's _custom_ops import then reads.
ensure_fork_mode_hook() {
  local sc=$V/lib/python3.12/site-packages/sitecustomize.py
  grep -q "rdna_fork_mode" "$sc" 2>/dev/null && return 0
  python3 - "$sc" <<'HOOKPY'
import sys
p = sys.argv[1]
block = (
    "_mode_file = os.path.expanduser(\"~/.cache/rdna_fork_mode\")\n"
    "if os.path.exists(_mode_file):\n"
    "    try:\n"
    "        with open(_mode_file) as _f:\n"
    "            for _line in _f:\n"
    "                _line = _line.strip()\n"
    "                if \"=\" in _line:\n"
    "                    _k, _v = _line.split(\"=\", 1)\n"
    "                    os.environ[_k.strip()] = _v.strip()\n"
    "    except Exception:\n"
    "        pass\n"
)
with open(p) as f:
    src = f.read()
out, done = [], False
for ln in src.split("\n"):
    out.append(ln)
    if not done and ln == "import os":
        out.append(block.rstrip("\n"))
        done = True
with open(p, "w") as f:
    f.write("\n".join(out))
HOOKPY
}

# Robust "ours" detection: catches vllm processes whose venv python is a
# symlink to /usr/bin/python3.12 (so a resolved-exe path match would miss
# them). Matches on the distinctive VLLM::*/PleOffloadWorker comm OR the venv
# path in the cmdline (api_server / resource_tracker / bench driver).
our_pids() {
  local p
  for p in $(pgrep -f "VLLM::Worker|VLLM::EngineCore|entrypoints.openai.api_server|entrypoints.cli.main serve|PleOffloadWorker|resource_tracker" 2>/dev/null); do
    if grep -qE "^VLLM::|^PleOffloadWorker" "/proc/$p/comm" 2>/dev/null; then
      echo "$p"
    elif tr '\0' '\n' < "/proc/$p/cmdline" 2>/dev/null | grep -q "venv-7.14.0_0.28.0"; then
      echo "$p"
    fi
  done
}

hygiene() {
  local n=0 left p
  for p in $(our_pids); do kill -TERM "$p" 2>/dev/null && n=$((n + 1)); done
  [ "$n" -gt 0 ] && sleep 8
  for p in $(our_pids); do kill -KILL "$p" 2>/dev/null; done
  sleep 2
  left=$(our_pids | tr '\n' ' ')
  log "hygiene: terminated=$n remaining_our=[${left% }]"
  [ -n "${left// /}" ] && return 1
  return 0
}

sel_count() { timeout 10 sudo -n ipmitool sel list 2>/dev/null | grep -c 'PCI SERR'; }
sel_snap() {
  local tag=$1 c
  c=$(sel_count)
  log "SEL[$tag] PCI-SERR=$c base=${SERR0:-?} uptime=$(uptime -p)"
  if [ -n "${SERR0:-}" ] && [ "${c:-0}" -gt "$SERR0" ]; then
    echo "STOP NEW PCI SERR after $tag ($c > $SERR0)" >"$OUT/STOP"
    log "STOP: new PCI SERR after $tag"; return 1
  fi
  return 0
}

teardown() {
  local api p
  api=$(grep -oE "APIServer pid=[0-9]+" "$D/serve.log" 2>/dev/null | head -1 | cut -d= -f2)
  [ -n "$api" ] && { kill -TERM "$api" 2>/dev/null; sleep 10; }
  for p in $(our_pids); do kill -TERM "$p" 2>/dev/null; done
  sleep 6
  for p in $(our_pids); do kill -KILL "$p" 2>/dev/null; done
  sleep 2
}

# Per-arm cell list.
cells_for_arm() {
  case "$1" in
    fp32-m0|fp32-m2) echo "1 16384 1024 111|8 16384 1024 112|1 1024 512 221|8 1024 512 222" ;;
    cas-m0|cas-m2)   echo "1 16384 1024 111|1 1024 512 221" ;;
    w4a8-m0)         echo "8 16384 1024 112" ;;
    *)               echo "" ;;
  esac
}

run_arm() {
  local arm=$1 FP32 W4A8 MTP ATTN PORT
  case "$arm" in
    fp32-m0) FP32=1 W4A8=0 MTP=0 ATTN=fa PORT=18410 ;;
    fp32-m2) FP32=1 W4A8=0 MTP=2 ATTN=fa PORT=18411 ;;
    cas-m0)  FP32=0 W4A8=0 MTP=0 ATTN=fa PORT=18412 ;;
    cas-m2)  FP32=0 W4A8=0 MTP=2 ATTN=fa PORT=18413 ;;
    w4a8-m0) FP32=1 W4A8=1 MTP=0 ATTN=fa PORT=18414 ;;
    *) log "unknown arm: $arm"; return 1 ;;
  esac
  local CELLS
  CELLS=$(cells_for_arm "$arm")
  local TAG=${TAG_OVERRIDE:-${TAG_PREFIX}_$arm}
  local CACHE=$T/cache/arm-fp32-$arm
  D=$OUT/$TAG
  mkdir -p "$D/cells"

  log "=== ARM=$arm FP32=$FP32 W4A8=$W4A8 MTP=$MTP ATTN=$ATTN PORT=$PORT HIP=$HIP_DEVICES TAG=$TAG ==="
  hygiene || { log "aborting: hygiene failed"; return 1; }
  if ss -ltn 2>/dev/null | grep -q ":$PORT "; then log "aborting: port $PORT bound"; return 1; fi
  rm -rf "$CACHE"; mkdir -p "$CACHE/inductor" "$CACHE/extensions"
  if [ -d "$TRITON_TEMPLATE" ]; then
    cp -a "$TRITON_TEMPLATE" "$CACHE/triton"
    log "triton cache seeded ($(ls "$CACHE/triton" | wc -l) entries)"
  else mkdir -p "$CACHE/triton"; fi
  rm -f "$CACHE/rdna_ar_wedged"
  SERR0=$(sel_count)
  log "arm cache=$CACHE baseline PCI-SERR=$SERR0"
  sel_snap preflight || return 2

  ROCK_MODE=$HOME/.cache/rdna_fork_mode
  mkdir -p "$HOME/.cache"
  printf 'VLLM_RDNA2_MOE_FP32_ACCUM=%s\n' "$FP32" > "$ROCK_MODE"
  [ "$W4A8" = "1" ] && printf 'VLLM_RDNA2_W4A8_SDOT4=1\n' >> "$ROCK_MODE"

  T0=$(date +%s)
  setsid nohup env \
    VENV="$V" VLLM_TREE="$T" MTP="$MTP" ATTN="$ATTN" TP=4 PORT="$PORT" MODEL="$MODEL" \
    VLLM_CACHE_ROOT="$CACHE" TORCHINDUCTOR_CACHE_DIR="$CACHE/inductor" \
    TRITON_CACHE_DIR="$CACHE/triton" TORCH_EXTENSIONS_DIR="$CACHE/extensions" \
    HIP_VISIBLE_DEVICES="$HIP_DEVICES" \
    VLLM_RDNA2_MOE_FP32_ACCUM="$FP32" \
    VLLM_RDNA2_W4A8_SDOT4="$W4A8" \
    VLLM_RDNA_AR=1 VLLM_RDNA_AR_MAX_KB=64 VLLM_RDNA_AR_ONESHOT_KB=64 \
    VLLM_RDNA_AR_BLOCKS=0 VLLM_RDNA_AR_PACE=0 VLLM_FORCE_CUSTOM_ALL_REDUCE=0 \
    NCCL_P2P_LEVEL=pxb RCCL_P2P_NET_DISABLE=1 RCCL_P2P_BATCH_ENABLE=1 \
    NCCL_PROTO=Simple RCCL_MSCCL_ENABLE=0 PYTHONFAULTHANDLER=1 \
    bash "$T/scripts/serve_gfx1030_flashnext_mtp.sh" >"$D/serve.log" 2>&1 </dev/null &
  log "launched arm=$arm cache=$CACHE log=$D/serve.log"

  local ready=0 dead=0 i
  for i in $(seq 1 120); do
    sleep 10
    curl -s --max-time 4 "http://127.0.0.1:$PORT/v1/models" 2>/dev/null | grep -q flash-next && { ready=1; break; }
    [ -f "$OUT/STOP" ] && break
    if pgrep -f "[a]pi_server" >/dev/null; then dead=0; else dead=$((dead + 1)); fi
    if [ "$dead" -ge 3 ]; then log "server gone during boot (after $((i*10))s)"; break; fi
  done
  local COLD=$(( $(date +%s) - T0 ))
  if [ "$ready" != "1" ]; then
    log "NOT READY after ${COLD}s"; tail -30 "$D/serve.log" | tee -a "$DRIVER_LOG" | tail -20
    teardown; hygiene; echo "NOT_READY" >"$D/status.txt"; return 1
  fi
  log "READY t=${COLD}s"; echo "$COLD" >"$D/cold_compile_seconds.txt"
  sel_snap after_boot || { teardown; hygiene; return 1; }

  # Warmup (throwaway) then coherence.
  for wl in "1024 64 999" "16384 64 998"; do
    set -- $wl
    ( cd "$D" && "$V/bin/python" -m vllm.entrypoints.cli.main bench serve \
      --backend openai --endpoint /v1/completions --base-url "http://127.0.0.1:$PORT" \
      --model "$MODEL" --served-model-name flash-next --dataset-name random \
      --random-input-len "$1" --random-output-len "$2" --num-prompts 1 --max-concurrency 1 \
      --ignore-eos --request-rate inf --seed "$3" --temperature 0 \
      --save-result --result-dir "$D/warmup_$1" ) >"$D/warmup_$1.log" 2>&1
  done
  log "warmup done"
  "$V/bin/python" "$T/tools/rdna2_028/probe_w4a8.py" "http://127.0.0.1:$PORT/v1" flash-next 1 >"$D/coherence.txt" 2>&1
  log "coherence rc=$? $(grep -c 'OK ' "$D/coherence.txt")/$(grep -cE '^\[' "$D/coherence.txt") OK"

  # Measured cells.
  local cell
  IFS='|' read -ra _cells <<<"$CELLS"
  for cell in "${_cells[@]}"; do
    set -- $cell
    local N=$1 IN=$2 OUTL=$3 SEED=$4
    local CD="$D/cells/c${IN}_${N}"
    log "bench c=$N in=$IN start"
    ( cd "$D" && "$V/bin/python" -m vllm.entrypoints.cli.main bench serve \
      --backend openai --endpoint /v1/completions --base-url "http://127.0.0.1:$PORT" \
      --model "$MODEL" --served-model-name flash-next --dataset-name random \
      --random-input-len "$IN" --random-output-len "$OUTL" --num-prompts "$N" --max-concurrency "$N" \
      --ignore-eos --request-rate inf --seed "$SEED" --temperature 0 \
      --save-result --result-dir "$CD" ) >"$CD.log" 2>&1
    log "bench c=$N in=$IN done: $(grep -m1 'Output token throughput' "$CD.log" | tr -s ' ')"
    "$V/bin/python" - "$PORT" "$CD" <<'PY' >"$CD/coherence.txt" 2>&1
import json, sys, urllib.request
port, out = sys.argv[1], sys.argv[2]
def ask(p, mt=16):
    body = json.dumps({"model": "flash-next", "prompt": p, "max_tokens": mt,
                       "temperature": 0.0, "ignore_eos": True}).encode()
    r = urllib.request.Request(f"http://127.0.0.1:{port}/v1/completions", data=body,
                               headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(r, timeout=600) as resp:
        return json.load(resp)["choices"][0]["text"] or ""
for name, p, exp in [("france", "The capital of France is", "paris"), ("math", "2 + 2 =", "4")]:
    try: t = ask(p)
    except Exception as e: t = f"<ERROR {e!r}>"
    print(f"[{name}] coherent={exp in t.lower()} | {t[:120]!r}")
PY
    sel_snap "after_c${N}_${IN}" || { teardown; hygiene; return 1; }
  done

  # Markers.
  {
    echo "=== W4A8 MoE marker (expected iff W4A8=1) ==="
    grep -m4 "W4A8\|w4a8\|sdot4" "$D/serve.log" | grep -v "W4A8-CALLTRACE" | head -8 || echo "(none)"
    echo "=== fp32 accum ==="
    grep -m2 "fp32_accum\|FP32_ACCUM\|fp32 accum" "$D/serve.log" | head -4 || echo "(none)"
    echo "=== TunableOp files ==="
    grep -h "reading tuning results from\|TunableOp lookup\|record" "$D/serve.log" | sort -u | head -6 || echo "(none)"
    echo "=== rdna_ar ==="
    grep -h "rdna_ar:" "$D/serve.log" | sort -u | head -4 || echo "(none)"
    echo "=== attention backend ==="
    grep -h "RDNA_ATTN\|TRITON_ATTN\|attention backend" "$D/serve.log" | sort -u | head -4 || echo "(none)"
  } >"$D/markers.txt" 2>&1
  tail -14 "$D/markers.txt" | tee -a "$DRIVER_LOG"

  "$V/bin/python" "$T/tools/rdna2_028/flashnext_w4a8_extract.py" "$D" >"$D/cells.csv" 2>"$D/extract.err" || true
  cat "$D/cells.csv" | tee -a "$DRIVER_LOG" | tail -8
  grep "SpecDecoding metrics" "$D/serve.log" >"$D/acceptance.txt" 2>/dev/null || true

  teardown
  hygiene
  sel_snap after_teardown || true
  rm -f "$ROCK_MODE"
  log "done $arm (FP32=$FP32 W4A8=$W4A8 MTP=$MTP ATTN=$ATTN)"
  echo "OK" >"$D/status.txt"
}

# --- Driver ----------------------------------------------------------------
ensure_fork_mode_hook
SERR0=$(sel_count)
log "=== fp32_accum_matrix start ARMS=[$ARMS] HIP=$HIP_DEVICES PCI-SERR=$SERR0 uptime=$(uptime -p) ==="
hygiene || { log "hygiene failed at start"; exit 1; }

rc=0
for arm in $ARMS; do
  log "---------------------------------------------------------------"
  if ! run_arm "$arm"; then
    log "ARM FAIL: $arm"
    rc=1
    grep -q "PCI SERR" "$OUT/STOP" 2>/dev/null && { log "stopping: PCI SERR"; break; }
    hygiene || { log "hygiene failed after $arm"; break; }
  fi
done

SERR1=$(sel_count)
log "=== fp32_accum_matrix done rc=$rc PCI-SERR=$SERR1 (base $SERR0) ==="
[ "$SERR1" -gt "$SERR0" ] && { echo "STOP NEW PCI SERR" >"$OUT/STOP"; log "STOP: new PCI SERR"; exit 2; }
echo "DONE rc=$rc" >"$OUT/fp32_accum_matrix.status"
exit $rc
