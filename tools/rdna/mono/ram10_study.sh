#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# TP4 R/F/D/Z expert-cache study: one boot per arm. R resident 0 slots (W4A8,
# mode 3); F fullcache 128; D demand 115 = 116 rows (12 saved, 89.84%); Z zeros
# 115 + constant zeros. Offload forces W4A8_SDOT4=0 and breakable graphs (mode
# NONE): backend/precision difference, not PCIe. LOAD_HOOK does reset ->
# exact-length warmup -> reset -> measured --save-detailed bench and keeps
# health, raw preempt metrics, KFD verdict, cache deltas, exact KV bytes.
# Overrides: ARM CELLS REPLICATE OUT_ROOT OUT TAG VENV MODEL PLE PLE_QUANT_DIR
# GPUS PORT TREE_SHA AFTER_MATRIX_HOOK REPORT RECIPE TP GMEM KV MAXLEN MAXBAT
# BLOCK_SIZE SEQS BENCH_EXTRA RESET_PREFIX_CACHE WARMUP SEED RESIDENT_SDOT4
# FORCE. CELLS in:out:conc[:requests]; none = probes. --dryrun/PRINT=1 prints
# launch/resets/bench argv, no GPU. BOOT_LOCK rechecked per cell; timeout
# fails. Quiet renews per cell, reopens after QUIET_WIN_MAX. KFD jump invalid,
# missing KFD/preemptions/KV RPC fail. Missing values stay null.
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
TREE=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
SELF=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")
REPORT_PY=$TREE/tools/rdna/mono/ram10_report.py
VALIDATE=$TREE/tools/rdna/port_v031/serve_validate.sh
OFFLOAD=$TREE/tools/rdna/mono/serve_offload.sh
KFD=$TREE/tools/rdna/port_v031/kfd_evictions.py
DEFAULT_MODEL=$HOME/.cache/huggingface/hub/models--wtdcode--Qwen3.8-Flash-Next-AWQ-W4A16/snapshots/0939125b929543a783ce700c90e36dd1a575c00c
DEFAULT_PLE=$HOME/.cache/huggingface/hub/models--primitive-ai--Qwen3.8-Flash-Next-PLE-quant/snapshots/4f861b63f69e61bfc2e22130ec91ec67f03ec43e/ples_int4
CPU_ENV=(env -u ROCR_VISIBLE_DEVICES -u CUDA_VISIBLE_DEVICES HIP_VISIBLE_DEVICES=)
QUIET_DIR=${QUIET_DIR:-$HOME/w4a8_runs/lanes/quiet-b4}
BOOT_LOCK=${BOOT_LOCK:-$HOME/w4a8_runs/boot.lock}

die() {
    printf 'ram10_study: %s\n' "$*" >&2
    exit 2
}
print_cmd() {
    local a
    for a in "$@"; do
        printf '%q ' "$a"
    done
    printf '\n'
}
arm_name() {
    case $1 in
        R) echo resident ;;
        F) echo fullcache ;;
        D) echo demand ;;
        Z) echo zeros ;;
    esac
}
arm_slots() {
    case $1 in
        R) echo 0 ;;
        F) echo 128 ;;
        D | Z) echo 115 ;;
    esac
}
normalize_arm() {
    case ${1,,} in
        r | resident) echo R ;;
        f | fullcache | full) echo F ;;
        d | demand) echo D ;;
        z | zeros | zero) echo Z ;;
        *) return 1 ;;
    esac
}

build_launch_plan() {
    ARM=$1
    ARM_NAME=$(arm_name "$1")
    SLOTS=$(arm_slots "$1")
    LOCAL_EXPERTS=128
    DUMMY=0
    DEVICE_ROWS=
    ROWS_SAVED=0
    RESIDENT_FRACTION=1.0
    if [[ $ARM == R ]]; then
        W4A8_PREFILL=$RESIDENT_SDOT4
        W4A8_SDOT4=$RESIDENT_SDOT4
        BREAKABLE=0
        COMPILE_MODE=3
        if [[ ${VLLM_USE_BREAKABLE_CUDAGRAPH:-} == 1 ]]; then
            BREAKABLE=1
            COMPILE_MODE=0
        fi
    else
        W4A8_PREFILL=0
        W4A8_SDOT4=0
        BREAKABLE=1
        COMPILE_MODE=0
        if ((SLOTS < LOCAL_EXPERTS)); then
            DUMMY=1
        fi
        DEVICE_ROWS=$((SLOTS + DUMMY))
        ROWS_SAVED=$((LOCAL_EXPERTS - DEVICE_ROWS))
        RESIDENT_FRACTION=$(awk -v s="$SLOTS" -v l="$LOCAL_EXPERTS" 'BEGIN { printf "%.9f", s / l }')
    fi
    MONO_CFG=
    if [[ $ARM == Z ]]; then
        MONO_CFG='"expert_cache_constant_qzeros":true'
    fi
    MONO_KNOBS=
    if [[ $ARM != R ]]; then
        MONO_KNOBS="{\"expert_cache_slots\":$SLOTS,\"expert_cache_policy\":\"$POLICY\",\"expert_cache_decay\":$DECAY${MONO_CFG:+,$MONO_CFG}}"
    fi
    TAG="${BASE_TAG}-${ARM_NAME}-r${REPLICATE}"
    RUN_OUT=$OUT_ROOT/$TAG
    if [[ -n ${OUT_OVERRIDE:-} ]]; then
        RUN_OUT=$OUT_OVERRIDE
        TAG=$(basename "$RUN_OUT")
    fi
    LAUNCH_ENV=(MODEL="$MODEL" VENV="$VENV" PLE="$PLE" PLE_QUANT_DIR="$PLE_QUANT_DIR"
        RECIPE="$RECIPE" GPUS="$GPUS" PORT="$PORT" TAG="$TAG" OUT="$RUN_OUT" CELLS=none
        LOAD_HOOK="$SELF" HIP_VISIBLE_DEVICES="$GPUS" VLLM_RDNA2_W4A16_DETERMINISTIC_DENSE=1
        VLLM_SERVER_DEV_MODE="${VLLM_SERVER_DEV_MODE:-1}" RAM10_LOAD_HOOK=1 RAM10_ARM="$ARM"
        RAM10_SLOTS="$SLOTS" RAM10_REPLICATE="$REPLICATE" RAM10_TAG="$TAG" RAM10_CELLS="$CELLS"
        RAM10_BENCH_EXTRA="$BENCH_EXTRA" RAM10_SEED="$SEED" RAM10_WARMUP="$WARMUP"
        RAM10_RESET_PREFIX="$RESET_PREFIX_CACHE" RAM10_VENV="$VENV" COMPILE_MODE="$COMPILE_MODE"
        RAM10_BASE_URL="http://127.0.0.1:$PORT")
    if ((BREAKABLE)); then
        LAUNCH_ENV+=(VLLM_USE_BREAKABLE_CUDAGRAPH=1)
    fi
    if [[ -n ${AFTER_MATRIX_HOOK:-} ]]; then
        LAUNCH_ENV+=("AFTER_MATRIX_HOOK=$AFTER_MATRIX_HOOK")
    fi
    if [[ $ARM == R ]]; then
        LAUNCH_ENV+=(EXTRA_ARGS="--monokernel --worker-extension-cls tools.rdna.mono.expert_cache_worker.ExpertCacheWorker")
        LAUNCH_ARGS=(TP="$TP" GMEM="$GMEM" KV="$KV" MAXLEN="$MAXLEN" MAXBAT="$MAXBAT"
            BLOCK_SIZE="$BLOCK_SIZE" SEQS="$SEQS" VLLM_RDNA2_W4A8_SDOT4="$RESIDENT_SDOT4"
            SERVED_NAME="$SERVED_NAME")
        LAUNCHER=$VALIDATE
    else
        LAUNCH_ENV+=(VALIDATE=1 TP="$TP" ALLOWED_GPU_SETS="$GPUS" SLOTS="$SLOTS" KV_OFFLOAD_GB=0
            MONO_CFG="$MONO_CFG" POLICY="$POLICY" DECAY="$DECAY" GMEM="$GMEM" MAXLEN="$MAXLEN"
            MAXBAT="$MAXBAT" EXTRA_ARGS="--worker-extension-cls tools.rdna.mono.expert_cache_worker.ExpertCacheWorker")
        LAUNCH_ARGS=(KV="$KV" BLOCK_SIZE="$BLOCK_SIZE" SEQS="$SEQS" VLLM_RDNA2_W4A8_SDOT4=0
            SERVED_NAME="$SERVED_NAME")
        LAUNCHER=$OFFLOAD
    fi
    LAUNCH_CMD=(env "${LAUNCH_ENV[@]}" bash "$LAUNCHER" "${LAUNCH_ARGS[@]}")
}

write_metadata() {
    printf '%s\0' "${LAUNCH_CMD[@]}" > "$RUN_OUT/launch.argv"
    "${CPU_ENV[@]}" "$VENV/bin/python" - "$RUN_OUT/metadata.json" "$1" \
        "arm=$ARM" "arm_name=$ARM_NAME" "replicate=$REPLICATE" "tag=$TAG" "cells=$CELLS" "slots=$SLOTS" \
        "model=$MODEL" "recipe=$RECIPE" "gpus=$GPUS" "tp=$TP" "gmem=$GMEM" "kv=$KV" "maxlen=$MAXLEN" \
        "maxbat=$MAXBAT" "seqs=$SEQS" "block_size=$BLOCK_SIZE" "out=$RUN_OUT" "tree=$TREE" "venv=$VENV" \
        "tree_sha=${TREE_SHA:-}" "ple=$PLE" "policy=$POLICY" "decay=$DECAY" "local_experts=$LOCAL_EXPERTS" \
        "dummy=$DUMMY" "device_rows=$DEVICE_ROWS" "rows_saved=$ROWS_SAVED" "resident_fraction=$RESIDENT_FRACTION" \
        "prefill=$W4A8_PREFILL" "sdot4=$W4A8_SDOT4" "compile_mode=$COMPILE_MODE" "breakable=$BREAKABLE" \
        "knobs=$MONO_KNOBS" "warmup=$WARMUP" "seed=$SEED" "bench_extra=$BENCH_EXTRA" "reset_prefix=$RESET_PREFIX_CACHE" <<'PY'
import glob, hashlib, json, os, subprocess, sys
from datetime import datetime, timezone
from pathlib import Path
meta, status, *pairs = sys.argv[1:]
e = dict(p.split("=", 1) for p in pairs)
def sha(p):
    try:
        with open(p, "rb") as f:
            return hashlib.file_digest(f, "sha256").hexdigest()
    except OSError:
        return None
def git(*a):
    try:
        return subprocess.run(
            ["git", "-C", e["tree"], *a],
            capture_output=True,
            text=True,
            timeout=60,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return None
def num(k, cast=int): return cast(e[k])
ts = e["tree_sha"] or (git("rev-parse", "HEAD") or "").strip() or None
dirty = git("status", "--porcelain") if not e["tree_sha"] else None
patterns = (
    e["tree"] + "/vllm/_rocm_C*.so",
    e["venv"] + "/lib/python*/site-packages/vllm/_rocm_C*.so",
    e["venv"] + "/vllm/_rocm_C*.so",
    e["tree"] + "/vllm/_C*.so",
    e["venv"] + "/lib/python*/site-packages/vllm/_C*.so",
    e["venv"] + "/vllm/_C*.so",
)
paths = list(
    {
        os.path.realpath(p): p
        for pattern in patterns
        for p in sorted(glob.glob(pattern))
    }.values()
)
hashes = [
    {"path": p, "size": os.path.getsize(p), "sha256": sha(p)}
    for p in paths
]
if hashes:
    kernel, kh = hashes[0]["path"], hashes[0]["sha256"]
else:
    kernel, kh = None, None
data = {
    "study": "ram10",
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "status": status,
    "arm": e["arm"],
    "arm_name": e["arm_name"],
    "replicate": e["replicate"],
    "tag": e["tag"],
    "cells": e["cells"].split(),
    "slots": num("slots"),
    "model": e["model"],
    "recipe": e["recipe"],
    "gpus": e["gpus"],
    "tp": num("tp"),
    "gmem": num("gmem", float),
    "kv": int(e["kv"]) if e["kv"].isdigit() else e["kv"],
    "maxlen": num("maxlen"),
    "maxbat": num("maxbat"),
    "seqs": num("seqs"),
    "block_size": num("block_size"),
    "source_sha": ts,
    "tree_sha": ts,
    "kernelSHA": kh,
    "kernel_sha256": kh,
    "kernel_path": kernel,
    "binary_hashes": hashes,
    "git_dirty": None if dirty is None else bool(dirty.strip()),
    "monokernel": True,
    "mtp": 2,
    "ple": e["ple"],
    "deterministic_dense": True,
    "w4a8_prefill": e["prefill"] == "1",
    "w4a8_sdot4": num("sdot4"),
    "compile_mode": num("compile_mode"),
    "cudagraph_mode": "FULL_AND_PIECEWISE",
    "breakable_cudagraph": e["breakable"] == "1",
    "expert_cache_policy": e["policy"],
    "expert_cache_decay": num("decay"),
    "mono_knobs": json.loads(e["knobs"]) if e["knobs"] else None,
    "local_experts": num("local_experts"),
    "device_cache_rows": int(e["device_rows"] or 0) or None,
    "rows_saved_vs_128": num("rows_saved"),
    "resident_fraction": (
        num("resident_fraction", float) if e["resident_fraction"] else None
    ),
    "kv_offload_gb": 0,
    "cpu_kv_tier": False,
    "eplb": int(os.environ.get("VLLM_RDNA_EPLB") or 0),
    "sdma": int(os.environ.get("HSA_ENABLE_SDMA") or 0),
    "gpu_max_hw_queues": int(os.environ.get("GPU_MAX_HW_QUEUES") or 4),
    "warmup_requests_per_cell": num("warmup"),
    "reset_prefix_cache": e["reset_prefix"] == "1",
    "bench_seed": num("seed"),
    "bench_extra": e["bench_extra"],
    "launch_argv": [
        p.decode(errors="replace")
        for p in Path(e["out"] + "/launch.argv").read_bytes().split(b"\0")
        if p
    ],
}
with open(meta, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2, sort_keys=True, allow_nan=False)
    f.write("\n")
PY
}

update_metadata() {
    "${CPU_ENV[@]}" "$VENV/bin/python" -c '
import json, sys
from datetime import datetime, timezone
p, r = sys.argv[1:]
with open(p) as f:
    d = json.load(f)
d.update(exit_code=int(r), status="done" if int(r) == 0 else "failed",
         finished_utc=datetime.now(timezone.utc).isoformat())
with open(p, "w") as f:
    json.dump(d, f, indent=2, sort_keys=True)
    f.write("\n")' \
        "$RUN_OUT/metadata.json" "$1"
}

# Load-hook side: HTTP helpers, quiet window, cells.
preempt_counter() {
    curl -fsS --max-time 30 "$BASE_URL/metrics" > "$1" || return 1
    awk '/^vllm:num_preemptions(_total)?(\{|[ \t])/ { s += $2; f = 1 }
        END { if (!f) exit 1; printf "%.0f\n", s }' "$1"
}
snapshot_cache_memory() {
    curl -sf --max-time 120 -X POST -H 'Content-Type: application/json' \
        -d '{"method":"rdna_cache_memory","kwargs":{}}' "$BASE_URL/collective_rpc" -o "$1" 2>/dev/null || return 1
    "${CPU_ENV[@]}" "$VENV/bin/python" -c '
import json, sys
with open(sys.argv[1]) as f:
    r = json.load(f).get("results")
ok = (
    isinstance(r, list)
    and len(r) == 4
    and all(
        isinstance(x.get("kv_cache_storage_bytes"), int)
        and x["kv_cache_storage_bytes"] > 0
        for x in r
    )
)
sys.exit(0 if ok else 1)
' "$1"
}
snapshot_cache() {
    if [[ $ARM == R ]]; then return 0; fi
    curl -sf --max-time 120 -X POST -H 'Content-Type: application/json' \
        -d '{"method":"rdna_expert_cache_control","kwargs":{}}' "$BASE_URL/collective_rpc" -o "$1" 2>/dev/null || { rm -f "$1"; return 1; }
}
server_alive() {
    local pid
    [[ -s $OUT/serve.pid ]] || return 1
    pid=$(cat "$OUT/serve.pid")
    [[ -n $pid ]] && kill -0 "$pid" 2>/dev/null
}
reset_prefix() {
    local dest=$1 _attempt
    for _attempt in 1 2 3 4 5; do
        if curl -sf --max-time 30 -X POST "$BASE_URL/reset_prefix_cache" -o "$dest" 2>/dev/null &&
            "${CPU_ENV[@]}" "$VENV/bin/python" -c '
import json, sys
with open(sys.argv[1]) as f:
    sys.exit(json.load(f).get("success") is not True)
' "$dest"; then
            return 0
        fi
        sleep 1
    done
    return 1
}

quiet_lock_wait() {
    local start=$(date +%s) waited=0
    while ! flock -n "$BOOT_LOCK" true 2>/dev/null; do
        if (($(date +%s) - start > ${QUIET_LOCK_WAIT:-600})); then
            echo "RAM10 quiet: $BOOT_LOCK still held; refusing measurement"
            return 1
        fi
        echo "RAM10 quiet: boot lock held; waiting"
        sleep "${QUIET_POLL:-30}"
        waited=1
    done
    ((waited)) && sleep "${QUIET_SETTLE:-30}" || true
}
quiet_close() {
    if [[ -n ${QUIET_PID:-} ]]; then
        kill "$QUIET_PID" 2>/dev/null || true
        wait "$QUIET_PID" 2>/dev/null || true
        QUIET_PID=
    fi
    [[ -n ${QUIET_FILE:-} ]] || return 0
    rm -f "$QUIET_FILE"
    QUIET_FILE=
    echo "RAM10 quiet: window closed"
}
quiet_open() {
    mkdir -p "$QUIET_DIR" || return 1
    local shell=$$
    while pgrep -f "${BUILDX_PAT:-^(/usr/sbin/|/usr/bin/)?docker buildx build}" > /dev/null 2>&1; do
        echo "RAM10 quiet: docker buildx build running; waiting"
        sleep "${QUIET_POLL:-30}"
    done
    QUIET_FILE=$QUIET_DIR/${QUIET_OWN:-ram10-$TAG}
    touch "$QUIET_FILE" || return 1
    QUIET_T0=$(date +%s)
    (
        stop=$((QUIET_T0 + ${QUIET_CAP:-900}))
        while sleep "${QUIET_TOUCH:-60}" && (($(date +%s) < stop)) && kill -0 "$shell" 2>/dev/null; do
            touch "$QUIET_FILE" 2>/dev/null || break
        done
    ) & QUIET_PID=$!
    quiet_lock_wait || return 1
    echo "RAM10 quiet: window open $QUIET_FILE"
}
quiet_renew() {
    if [[ -n ${QUIET_FILE:-} ]] && (($(date +%s) - QUIET_T0 >= ${QUIET_WIN_MAX:-840})); then
        quiet_close
        echo "RAM10 quiet: gap ${QUIET_GAP:-120}s"
        sleep "${QUIET_GAP:-120}"
        quiet_open || return 1
    fi
    quiet_lock_wait
}

parse_cell() {
    IFS=: read -r CELL_INPUT CELL_OUTPUT CELL_CONC CELL_REQUESTS <<< "$1"
    CELL_NAME=bench-${CELL_INPUT}x${CELL_OUTPUT}-c${CELL_CONC}
    if [[ -z ${CELL_REQUESTS:-} ]]; then
        CELL_REQUESTS=$((CELL_CONC * 4 > 8 ? CELL_CONC * 4 : 8))
        if ((CELL_CONC == 1)); then
            CELL_REQUESTS=4
        fi
    fi
}
build_bench_argv() {
    local -a extra=()
    read -r -a extra <<< "${BENCH_EXTRA:-}"
    BENCH_ARGS=("$VENV/bin/python" -m vllm.entrypoints.cli.main bench serve
        --backend openai --endpoint /v1/completions --base-url "$BASE_URL"
        --model "$SERVED" --tokenizer "$MODEL" --dataset-name random
        --random-input-len "$1" --random-output-len "$2" --num-prompts "$3"
        --max-concurrency "$4" --ignore-eos --request-rate inf --seed "$SEED"
        --save-result --result-dir "$5" --result-filename "$6")
    BENCH_ARGS+=("${extra[@]}")
    BENCH_ARGS+=(--ready-check-timeout-sec 0 --num-warmups 0)
    BENCH_ARGS+=(--metadata "ram10_arm=$ARM" "ram10_slots=$SLOTS" "ram10_replicate=$REPLICATE"
        "ram10_cell=$CELL_NAME" "ram10_tag=$TAG" "ram10_role=$7")
}
run_bench() {
    build_bench_argv "$1" "$2" "$3" "$4" "$5" "$6" "$7"
    HIP_VISIBLE_DEVICES="$GPUS" "${BENCH_ARGS[@]}" > "$5/$7.log" 2>&1
}
write_cell() {
    cat > "$dir/cell.json" <<EOF
{"arm": "$ARM", "replicate": "$REPLICATE", "cell": "$CELL_NAME", "in": $CELL_INPUT,
 "out": $CELL_OUTPUT, "conc": $CELL_CONC, "requests": $CELL_REQUESTS, "bench_rc": $brc,
 "warmup_rc": $wrc, "warmup_enabled": $((WARMUP != 0)), "reset_before_warmup_ok": $rb,
 "reset_after_warmup_ok": $ra, "evictions": "$ev", "preemptions_in_window": $pre,
 "faults_in_window": $faults, "server_alive_after": $alive, "result_present": $rp,
 "warmup_present": $wp, "valid": false, "invalid_reasons": [], "warnings": []}
EOF
}
finalize_cell() {
    [[ -x $VENV/bin/python && -r $REPORT_PY ]] ||
        { echo "RAM10 CELL $CELL_NAME FUZZY finalizer unavailable"; return 2; }
    "${CPU_ENV[@]}" "$VENV/bin/python" "$REPORT_PY" --finalize-cell "$1" > "$1/finalize.json" 2>&1
}

run_cell() {
    parse_cell "$1"
    local dir=$OUT/cells/$CELL_NAME
    local pid="" frc=0 span=0 rb=1 ra=1 ok=1 brc=null wrc=null wp=0 rp=0
    local pre=null pb=0 pa=0 faults=null ev=unavailable alive=0 ls=0 le=0
    mkdir -p "$dir"
    LAST_CELL_VALID=0
    CELL_HARD=0
    hard_fail() {
        if server_alive; then
            alive=1
        fi
        write_cell
        echo "RAM10 CELL $CELL_NAME FAIL $1"
        CELL_HARD=1
        return 0
    }
    kfd() {
        "${CPU_ENV[@]}" "$VENV/bin/python" "$KFD" "$pid" "$@"
    }
    echo "RAM10 CELL $CELL_NAME in=$CELL_INPUT out=$CELL_OUTPUT conc=$CELL_CONC requests=$CELL_REQUESTS"
    if ((RESET_PREFIX)); then
        if ! reset_prefix "$dir/reset-before-warmup.json"; then
            rb=0
            ok=0
        fi
    fi
    if ((WARMUP && ok)); then
        wrc=0
        run_bench "$CELL_INPUT" "$CELL_OUTPUT" 1 1 "$dir" warmup.json warmup || wrc=$?
        if [[ -r $dir/warmup.json ]]; then
            wp=1
        fi
    fi
    if ((RESET_PREFIX)); then
        if ! reset_prefix "$dir/reset-after-warmup.json"; then
            ra=0
            ok=0
        fi
    fi
    if ((!ok)); then
        if server_alive; then
            alive=1
        fi
        write_cell
        finalize_cell "$dir" || true
        echo "RAM10 CELL $CELL_NAME FAIL prefix reset"
        CELL_HARD=1
        return 0
    fi
    snapshot_cache "$dir/cache-before.json" ||
        echo "RAM10 CELL $CELL_NAME warning: cache counters unavailable"
    if [[ -s $OUT/serve.pid ]]; then
        pid=$(cat "$OUT/serve.pid")
    fi
    if [[ -z $pid ]] || ! kfd --save "$dir/evicted-before.json" > /dev/null 2>&1; then
        hard_fail "KFD snapshot unavailable"
        return 0
    fi
    ls=$(wc -l < "$OUT/serve.log" 2>/dev/null || echo 0)
    pb=$(preempt_counter "$dir/metrics-before.txt") || {
        hard_fail "preemption metric missing"
        return 0
    }
    brc=0
    run_bench "$CELL_INPUT" "$CELL_OUTPUT" "$CELL_REQUESTS" "$CELL_CONC" "$dir" bench.json measured || brc=$?
    if [[ -r $dir/bench.json ]]; then
        rp=1
    fi
    le=$(wc -l < "$OUT/serve.log" 2>/dev/null || echo 0)
    pa=$(preempt_counter "$dir/metrics-after.txt") || {
        hard_fail "preemption metric missing"
        return 0
    }
    ((pa >= pb)) || {
        hard_fail "preemption counter reset"
        return 0
    }
    pre=$((pa - pb))
    snapshot_cache "$dir/cache-after.json" ||
        echo "RAM10 CELL $CELL_NAME warning: cache counters unavailable"
    span=$((le - ls))
    if ((span > 0)); then
        faults=$(tail -n +$((ls + 1)) "$OUT/serve.log" | head -n "$span" |
            grep -acE 'Memory Fault|HSA_STATUS_ERROR|page fault|Segmentation fault|core dumped|GPU Hang|HW Exception' || true)
        [[ -n $faults ]] || faults=0
    else
        faults=0
    fi
    if [[ -n $pid && -r $dir/evicted-before.json ]]; then
        kfd --since "$dir/evicted-before.json" > "$dir/evictions.txt" 2>&1 || true
        if grep -q 'INVALID' "$dir/evictions.txt" 2>/dev/null; then
            ev=INVALID
        elif grep -q -- '-> ok' "$dir/evictions.txt" 2>/dev/null; then
            ev=ok
        fi
    else
        echo "evictions unavailable" > "$dir/evictions.txt"
    fi
    if server_alive; then
        alive=1
    fi
    write_cell
    finalize_cell "$dir" || frc=$?
    if ((frc == 0)); then
        LAST_CELL_VALID=1
        echo "RAM10 CELL $CELL_NAME PASS"
    elif ((frc == 1)); then
        echo "RAM10 CELL $CELL_NAME INVALID"
        CELL_HARD=1
    else
        echo "RAM10 CELL $CELL_NAME FINALIZE_FAIL"
        CELL_HARD=1
        return 0
    fi
    if ((!alive)); then
        CELL_HARD=1
    fi
    return 0
}

load_hook_main() {
    : "${OUT:?LOAD_HOOK requires OUT}" "${PORT:?}" "${SERVED:?}" "${MODEL:?}"
    VENV=${RAM10_VENV:-${VENV:-$TREE/.venv}}
    ARM=${RAM10_ARM:?LOAD_HOOK requires RAM10_ARM}
    SLOTS=${RAM10_SLOTS:-0}
    REPLICATE=${RAM10_REPLICATE:-1}
    TAG=${RAM10_TAG:-ram10}
    SEED=${RAM10_SEED:-12345}
    WARMUP=${RAM10_WARMUP:-1}
    RESET_PREFIX=${RAM10_RESET_PREFIX:-1}
    BENCH_EXTRA=${RAM10_BENCH_EXTRA:-'--temperature 0 --save-detailed'}
    BASE_URL=${RAM10_BASE_URL:-${URL:-http://127.0.0.1:$PORT}}
    BASE_URL=${BASE_URL%/v1/completions}
    local specs=${RAM10_CELLS:-} hard=0 invalid=0 spec
    if [[ -z $specs ]]; then
        echo "RAM10 STUDY SKIP no cells"
        echo "RAM10 STUDY PASS"
        return 0
    fi
    if [[ ! -x $VENV/bin/python || ! -r $REPORT_PY ]]; then
        echo "RAM10 STUDY FAIL python/report unavailable"
        return 1
    fi
    QUIET_FILE=
    QUIET_PID=
    trap quiet_close EXIT
    if ! quiet_open; then
        echo "RAM10 STUDY FAIL quiet protocol"
        return 1
    fi
    rm -f "$OUT/cache-memory.json"
    if ! snapshot_cache_memory "$OUT/cache-memory.json"; then
        echo "RAM10 CACHE_MEMORY FAIL exact KV bytes unavailable"
        return 1
    fi
    for spec in $specs; do
        if ! quiet_renew; then
            hard=1
            break
        fi
        run_cell "$spec"
        if ((CELL_HARD)); then
            hard=1
            echo "RAM10 CELL $CELL_NAME HARD_FAIL"
            break
        fi
        ((LAST_CELL_VALID)) || invalid=1
    done
    quiet_close
    if [[ -n ${AFTER_MATRIX_HOOK:-} ]]; then
        if ((hard == 0 && invalid == 0)); then
            local hook=$AFTER_MATRIX_HOOK after_rc=0
            if [[ $hook != /* ]]; then
                hook=$TREE/$hook
            fi
            URL=${URL:-$BASE_URL/v1/completions}
            GPUS=${GPUS:-}
            export OUT URL PORT SERVED MODEL VENV GPUS TREE ARM
            echo "RAM10 AFTER_MATRIX $hook"
            if [[ -r $hook ]]; then
                bash "$hook" > "$OUT/after_matrix.log" 2>&1 || after_rc=$?
            else
                echo "RAM10 AFTER_MATRIX hook unreadable: $hook" | tee -a "$OUT/after_matrix.log"
                after_rc=1
            fi
            if ((after_rc != 0)); then
                echo "RAM10 AFTER_MATRIX FAIL rc=$after_rc"
                hard=1
            else
                echo "RAM10 AFTER_MATRIX PASS"
            fi
        else
            echo "RAM10 AFTER_MATRIX SKIP invalid matrix"
        fi
    fi
    if ((hard)); then
        echo "RAM10 STUDY FAIL hard cell failure"
        return 1
    fi
    if ((invalid)); then
        echo "RAM10 STUDY FAIL invalid cells"
        return 1
    fi
    echo "RAM10 STUDY PASS"
    return 0
}

print_plan() {
    echo "RAM10 STUDY PLAN"
    echo "  TREE=$TREE VENV=$VENV MODEL=$MODEL PLE=$PLE ($PLE_QUANT_DIR)"
    echo "  GPUS=$GPUS TP=$TP PORT=$PORT GMEM=$GMEM KV=$KV MAXLEN=$MAXLEN MAXBAT=$MAXBAT BLOCK_SIZE=$BLOCK_SIZE SEQS=$SEQS"
    echo "  CELLS=${CELLS:-<none: probes only>} REPLICATE=$REPLICATE"
    local arm spec name
    BASE_URL=http://127.0.0.1:$PORT
    SERVED=$SERVED_NAME
    for arm in "${ARMS[@]}"; do
        build_launch_plan "$arm"
        echo
        echo "ARM $ARM ($ARM_NAME) slots=$SLOTS dummy=$DUMMY rows=${DEVICE_ROWS:-resident} saved=$ROWS_SAVED"
        echo "  W4A8_PREFILL=$W4A8_PREFILL W4A8_SDOT4=$W4A8_SDOT4 BREAKABLE=$BREAKABLE COMPILE_MODE=$COMPILE_MODE"
        echo "  run=$RUN_OUT"
        echo -n "  LAUNCH "
        print_cmd "${LAUNCH_CMD[@]}"
        for spec in ${CELLS:-}; do
            parse_cell "$spec"
            name=$CELL_NAME
            echo "  CELL $spec requests=$CELL_REQUESTS"
            if ((RESET_PREFIX_CACHE)); then
                printf '    RESET curl -sf --max-time 30 -X POST %q -o %q\n' \
                    "http://127.0.0.1:$PORT/reset_prefix_cache" "$RUN_OUT/cells/$name/reset-before-warmup.json"
            fi
            if ((WARMUP)); then
                build_bench_argv "$CELL_INPUT" "$CELL_OUTPUT" 1 1 "$RUN_OUT/cells/$name" warmup.json warmup
                echo -n "    WARMUP "
                print_cmd "${BENCH_ARGS[@]}"
            fi
            if ((RESET_PREFIX_CACHE)); then
                printf '    RESET curl -sf --max-time 30 -X POST %q -o %q\n' \
                    "http://127.0.0.1:$PORT/reset_prefix_cache" "$RUN_OUT/cells/$name/reset-after-warmup.json"
            fi
            build_bench_argv "$CELL_INPUT" "$CELL_OUTPUT" "$CELL_REQUESTS" "$CELL_CONC" "$RUN_OUT/cells/$name" bench.json measured
            echo -n "    MEASURED "
            print_cmd "${BENCH_ARGS[@]}"
        done
    done
    if [[ -n ${OUT_OVERRIDE:-} ]]; then
        echo "REPORT SKIP single OUT; run ram10_report.py --run-root explicitly"
    else
        echo "REPORT $VENV/bin/python $REPORT_PY --run-root $OUT_ROOT --baseline R"
    fi
}

if [[ ${RAM10_LOAD_HOOK:-0} == 1 ]]; then
    load_hook_main
    exit $?
fi
DRY=0
OUT_OVERRIDE=
for arg in "$@"; do
    case $arg in
        --dryrun | --dry-run | -n | PRINT=1) DRY=1 ;;
        -h | --help) sed -n 's/^#\([^!].*\)/\1/p' "${BASH_SOURCE[0]}" >&2; exit 0 ;;
        *=*)
            key=${arg%%=*} value=${arg#*=}
            case $key in
                ARM | CELLS | REPLICATE | TAG | OUT_ROOT | OUT | PORT | TP | GPUS | GPU | VENV | MODEL | PLE | PLE_QUANT_DIR | RECIPE | GMEM | KV | MAXLEN | MAXBAT | BLOCK_SIZE | SEQS | BENCH_EXTRA | RESET_PREFIX_CACHE | WARMUP | SEED | RESIDENT_SDOT4 | FORCE | REPORT | TREE_SHA | AFTER_MATRIX_HOOK | POLICY | DECAY)
                    printf -v "$key" '%s' "$value" ;;
                *) die "unknown override $key" ;;
            esac ;;
        *) die "unexpected argument '$arg'" ;;
    esac
done
if [[ ${PRINT:-0} == 1 ]]; then
    DRY=1
fi
ARM=${ARM:-matrix}
CELLS=${CELLS:-'1024:512:1:4 1024:512:8:8 16384:1024:1:2 16384:1024:8:8 32768:2048:1:2 32768:2048:8:8'}
if [[ $CELLS == none ]]; then
    CELLS=
fi
REPLICATE=${REPLICATE:-1}
BASE_TAG=${TAG:-ram10}
OUT_ROOT=${OUT_ROOT:-$HOME/w4a8_runs/ram10-study}
PORT=${PORT:-18766}
TP=${TP:-4}
GPUS=${GPUS:-${GPU:-2,3,4,5}}
VENV=${VENV:-$TREE/.venv}
MODEL=${MODEL:-$DEFAULT_MODEL}
PLE=${PLE:-int4}
PLE_QUANT_DIR=${PLE_QUANT_DIR:-$DEFAULT_PLE}
RECIPE=${RECIPE:-flashnext-mtp2}
SERVED_NAME=${SERVED_NAME:-flash-next}
GMEM=${GMEM:-0.92}
KV=${KV:-auto}
MAXLEN=${MAXLEN:-131072}
MAXBAT=${MAXBAT:-2048}
BLOCK_SIZE=${BLOCK_SIZE:-1024}
SEQS=${SEQS:-8}
WARMUP=${WARMUP:-1}
SEED=${SEED:-12345}
BENCH_EXTRA=${BENCH_EXTRA:-'--temperature 0 --save-detailed'}
RESET_PREFIX_CACHE=${RESET_PREFIX_CACHE:-1}
RESIDENT_SDOT4=${RESIDENT_SDOT4:-1}
POLICY=${POLICY:-lfu_lru}
DECAY=${DECAY:-128}
FORCE=${FORCE:-0}
REPORT=${REPORT:-1}
if [[ -n ${OUT:-} ]]; then
    OUT_OVERRIDE=$OUT
fi
ARMS=()
if [[ ${ARM,,} == matrix || ${ARM,,} == all ]]; then
    ARMS=(R F D Z)
else
    IFS=',' read -r -a _items <<< "$ARM"
    for _item in "${_items[@]}"; do
        _arm=$(normalize_arm "$_item") || die "unknown ARM=$_item"
        case " ${ARMS[*]:-} " in
            *" $_arm "*) ;;
            *) ARMS+=("$_arm") ;;
        esac
    done
    ((${#ARMS[@]})) || die "no arms selected"
fi
if [[ $TP != 4 ]]; then
    die "this study is TP4-only (128 local experts/rank)"
fi
if [[ ! $GPUS =~ ^[0-9]+(,[0-9]+)*$ ]]; then
    die "GPUS must be a comma-separated HIP list"
fi
IFS=',' read -r -a _gpus <<< "$GPUS"
if ((${#_gpus[@]} != TP)); then
    die "GPUS lists ${#_gpus[@]} cards but TP=$TP"
fi
if [[ ! $PORT =~ ^[0-9]+$ ]] || ((PORT < 1 || PORT > 65535)); then
    die "PORT must be 1..65535"
fi
if [[ ! $REPLICATE =~ ^[A-Za-z0-9._-]+$ || ! $BASE_TAG =~ ^[A-Za-z0-9._-]+$ ]]; then
    die "invalid REPLICATE/TAG"
fi
if [[ ! $GMEM =~ ^[0-9]+([.][0-9]+)?$ ]]; then
    die "GMEM must be numeric"
fi
for _var in MAXLEN MAXBAT BLOCK_SIZE SEQS SEED DECAY; do
    if [[ ! ${!_var} =~ ^[0-9]+$ ]]; then
        die "$_var must be an integer"
    fi
done
if ((MAXLEN <= 0 || MAXBAT <= 0 || BLOCK_SIZE <= 0 || SEQS <= 0 || DECAY <= 0)); then
    die "invalid memory/context sizes"
fi
if [[ $KV != auto && ! $KV =~ ^[0-9]+$ ]]; then
    die "KV must be auto or bytes"
fi
if [[ $POLICY != lfu_lru && $POLICY != lru ]]; then
    die "POLICY must be lfu_lru or lru"
fi
for _var in RESIDENT_SDOT4 RESET_PREFIX_CACHE WARMUP FORCE REPORT; do
    if [[ ${!_var} != 0 && ${!_var} != 1 ]]; then
        die "$_var must be 0 or 1"
    fi
done
if [[ $BENCH_EXTRA != *--save-detailed* || ! $BENCH_EXTRA =~ (^|[[:space:]])--temperature[[:space:]]+0([[:space:]]|$) ]]; then
    die "BENCH_EXTRA needs --save-detailed and --temperature 0"
fi
if [[ $RESET_PREFIX_CACHE == 1 && ${VLLM_SERVER_DEV_MODE:-1} == 0 ]]; then
    die "RESET_PREFIX_CACHE=1 needs VLLM_SERVER_DEV_MODE=1"
fi
if [[ ${VLLM_USE_BREAKABLE_CUDAGRAPH:-} == 0 ]]; then
    die "offload arms need breakable graphs; unset VLLM_USE_BREAKABLE_CUDAGRAPH"
fi
CELL_SPECS=()
declare -A _seen_cells=()
for _spec in $CELLS; do
    if [[ ! $_spec =~ ^([0-9]+):([0-9]+):([0-9]+)(:([0-9]+))?$ ]]; then
        die "bad CELLS entry '$_spec'"
    fi
    _in=${BASH_REMATCH[1]}
    _out=${BASH_REMATCH[2]}
    _conc=${BASH_REMATCH[3]}
    _nreq=${BASH_REMATCH[5]:-}
    if ((_in <= 0 || _out <= 0 || _conc <= 0 || _conc > SEQS || ${_nreq:-1} <= 0)); then
        die "bad CELLS entry/concurrency '$_spec'"
    fi
    if ((_in + _out > MAXLEN)); then
        die "cell $_in+$_out exceeds MAXLEN=$MAXLEN"
    fi
    _name=bench-${_in}x${_out}-c${_conc}
    if [[ -n ${_seen_cells[$_name]:-} ]]; then
        die "duplicate cell '$_name'"
    fi
    _seen_cells[$_name]=1
    CELL_SPECS+=("$_spec")
done
CELLS="${CELL_SPECS[*]:-}"
if ((DRY)); then
    print_plan
    exit 0
fi
if [[ ! -r $VALIDATE || ! -r $OFFLOAD || ! -r $KFD || ! -r $REPORT_PY ]]; then
    die "harness/study files missing under $TREE"
fi
if [[ ! -x $VENV/bin/python ]]; then
    die "VENV=$VENV has no bin/python"
fi
overall=0
for _arm in "${ARMS[@]}"; do
    build_launch_plan "$_arm"
    if [[ -e $RUN_OUT && -n $(ls -A "$RUN_OUT" 2>/dev/null) && $FORCE != 1 ]]; then
        die "$RUN_OUT is not empty; set REPLICATE=<new> or FORCE=1"
    fi
    mkdir -p "$RUN_OUT"
    write_metadata running
    echo "RAM10 ARM $ARM ($ARM_NAME) slots=$SLOTS -> $RUN_OUT"
    set +e
    "${LAUNCH_CMD[@]}" 2>&1 | tee -a "$RUN_OUT/runner.log"
    rc=${PIPESTATUS[0]}
    set -e
    if ((rc == 0)); then
        if ! grep -qx 'RAM10 STUDY PASS' "$RUN_OUT/load_hook.txt" 2>/dev/null ||
            ! grep -q 'PROBE-AFTER-LOAD PASS' "$RUN_OUT/summary.txt" 2>/dev/null ||
            grep -qE 'PROBE FAIL|PPL PROBE FAIL|PREFIX FAIL|PREFIX RESET FAILED|CONCURRENCY FAIL|PROBE-AFTER-LOAD FAIL' "$RUN_OUT/summary.txt" 2>/dev/null; then
            echo "RAM10 qualification or load hook failed despite harness exit 0"
            rc=1
        fi
    fi
    update_metadata "$rc"
    if ((rc != 0)); then
        echo "RAM10 ARM $ARM FAIL rc=$rc"
        overall=$rc
        break
    fi
    echo "RAM10 ARM $ARM DONE"
done
if ((overall == 0)) && [[ $REPORT == 1 ]]; then
    if [[ -n $OUT_OVERRIDE ]]; then
        echo "RAM10 REPORT SKIP single OUT; run ram10_report.py --run-root explicitly"
    else
        echo "RAM10 REPORT $OUT_ROOT"
        "${CPU_ENV[@]}" "$VENV/bin/python" "$REPORT_PY" --run-root "$OUT_ROOT" --baseline R || overall=$?
    fi
fi
exit "$overall"
