#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# ONE detached, resumable, stop-on-first-failure validation of the final HI2
# Inkling-2-Mini card on the a2 lease (HIP 0,1; PCI d1 d4, TP=2).
#
#   bash tools/inkling_hi2/validate_final_a2.sh MODEL=~/w4a8_runs/inkling-hi2/stage4/runs/bpw3.5
#   TAG=mytag bash tools/inkling_hi2/validate_final_a2.sh MODEL=...   # resume that run
#   ONLY=kernel,serve_dir ...                                        # rerun a subset
#   DRY_RUN=1 ...                                                    # argv check only, no lease
#
# Resumable: every PASSed step is a marker under $RUN/steps/; rerun the same TAG
# (or RUN=) and the step is skipped.  A failed step is retried; nothing after it
# ran.  Every boot goes through serve.sh (lease request/holder, 28 GiB gate, the
# slot's quiet windows, the cross-slot boot.lock, the HIP->PCI litmus); the
# kernel test runs in its own lease window with the same quiet/boot-lock rule
# (it has no server to carry the lease).  After it, the driver takes ONE lease for
# the whole run and every boot it starts inherits it (LEASE_INHERIT=1: the boot
# verifies the holder is ours+alive and the acquisition-time 28 GiB gate, writes
# no nested request, and never releases it); a foreign holder still refuses.
# All GPU commands are dry-run (argv only, CPU, no lease) before the first lease
# is taken.
#
# Results: $RUN (default ~/w4a8_runs/inkling-hi2-runtime/final/<UTC tag>):
#   STATUS.log  -- timestamped BEGIN/SKIP/PASS/FAIL lines
#   steps/NAME.log, steps/NAME.json (rc, seconds, numbers), steps/NAME.pass
#   logs/  -- server/instrument logs; pids/ -- SERVER_PID_FILE paths
#   dryrun.log -- every GPU command's argv, checked before the lease
#   summary.md -- PASS/FAIL per step with the numbers (written at the end or on
#                 the first failure)
#
# Step order (grouped to minimise boots; all requested checks are here):
#   1 kernel      (lease window) HI2 kernel unit test on the merged tree
#   2 serve_dir   prep_serve_dir + validate_dir --require-mtp + check_real_dir
#                 --vision --mtp (CPU)
#   3 smoke       stock resident, smoke.sh boots its own HI2 (two-server caps)
#   4 kl          stock+HI2 pair: mixed 32 windows + long 8k/16k (8 each,
#                 max_model_len 32768); mixed gated at 0.15, long recorded
#   5 eval        pair: task eval main 245 + agentic-probe + vision, labels
#                 stock, stock2, hi2, then compare (eval tree at EVAL_TREE)
#   6 vision      pair (VISION=1): vision_probe.py + its 4-way mixed batch
#   7 bench       pair arm (GMEM 0.45 each) then a SOLO arm: stock/HI2 stopped,
#                 HI2 alone at GMEM 0.92 with the recipe's MAXBAT/MAXLEN; both
#                 arms use --decode-tokens 128 and both are recorded in the
#                 step JSON with their flags (the card quotes the solo arm)
#   7b bench8     `vllm bench serve` over the HTTP API (dataset-name random):
#                 stock solo then HI2 solo, ctx 32768, SEQS=8, MAXBAT=2048,
#                 cells 1024->512 n=16 and 16384->512 n=8 (concurrency 8);
#                 each cell must complete num-prompts with failed=0, the
#                 server healthy before/after, and the serve log showing the
#                 FULL_AND_PIECEWISE graph capture; results JSON kept per cell
#   8 boot720     single HI2 DSPARK=1, 720000 context (GMEM=0.92) + health
#                 (before dspark: a DSpark verdict must not block the context
#                 test)
#   9 dspark      HI2 spec + TWO plain boots (16384): the cross-process floor,
#                 identity within it, per-position, no-draft batch 3; then a
#                 stock DSPARK=1 boot for the acceptance reference
#
# Estimated a2 wall time: about 4 h, budget 5 h (do not start with < 5 h free):
# CPU/preflight 15 min; kernel 3; smoke 6; KL 25-35; bench 12-18; task eval
# 70-90 (main 245 items at 4 workers: stock ~10 min, HI2 ~15 min; agentic and
# vision sets and the three compares on top); vision probe 6; DSpark 12; 720k
# boot 8; boots/loads ~25; lease/quiet waits and retries on top of that.
#
# Exit codes: 0 all steps PASS, 1 a step failed (summary written, servers
# stopped), 2 bad usage, 3 preflight/dry-run/guard failure (no lease taken).
set -uo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
TREE=${TREE:-$(cd "$HERE/../.." && pwd)}
RT=${RT:-$HOME/w4a8_runs/inkling-hi2-runtime}

MODEL=${MODEL:-${1:-}}
if [[ -z ${MODEL:-} || ! -d $MODEL ]]; then
    echo "validate_final_a2: MODEL must point at the quantizer output directory" >&2
    exit 2
fi
MODEL=$(cd "$MODEL" && pwd -P)

SLOT=${SLOT:-a2}
case $SLOT in
    a2) DEF_HIP=0,1 DEF_BUSES=d1,d4 ;;
    *)
        echo "validate_final_a2: TP=2 belongs on a2 (HIP 0,1; PCI d1 d4)" >&2
        exit 2
        ;;
esac
HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-$DEF_HIP}
EXPECT_PCI_BUSES=${EXPECT_PCI_BUSES:-$DEF_BUSES}
OWNER=${OWNER:-HI2InklingRuntime}
LEASE_DIR=${LEASE_DIR:-$HOME/w4a8_runs/lanes/lease}
QUIET_DIR=${QUIET_DIR:-$HOME/w4a8_runs/lanes/quiet-$SLOT}
BOOT_LOCK=${BOOT_LOCK:-$HOME/w4a8_runs/boot.lock}
LEASE_FREE_GIB=${LEASE_FREE_GIB:-28}

V=${V:-$HOME/Apps/vllm/venv-7.14.0_0.34.0-rebase-1e1cac5340/bin/python}
VENV=${VENV:-$(dirname "$(dirname "$V")")}
QUANTIZER_TREE=${QUANTIZER_TREE:-$HOME/w4a8_runs/inkling-hi2/tree}
DEPS=${DEPS:-$HOME/w4a8_runs/pydeps/tfm-main}
TINY=${TINY:-$RT/tiny}

STOCK=${STOCK:-$HOME/hfcache/inkling2-mini}
SERVE_DIR=${SERVE_DIR:-$RT/serve-$(basename "$MODEL")}
SERVE=${SERVE:-$HERE/serve.sh}
SMOKE=${SMOKE:-$HERE/smoke.sh}
PREP=${PREP:-$HERE/prep_serve_dir.sh}
VALIDATE=${VALIDATE:-$HERE/validate_dir.py}
CHECK_REAL=${CHECK_REAL:-$HERE/check_real_dir.py}
KLPROBE=${KLPROBE:-$HERE/kl_probe.py}
BENCH=${BENCH:-$HERE/bench.py}
# bench8 uses the HTTP client, not the tree's in-process bench: the runtime
# venv's `import vllm` aborts on this box with an LLVM "spirv-expand-step"
# duplicate-option error (verified 2026-11-26, with GPUs visible); the 0.32
# venv's `vllm bench serve` imports and prints usage, and its result JSON
# schema (percentiles_*/median_*/completed/output_throughput) is the one
# b8_extract parses.  Set BENCH8_PY to another interpreter if that changes.
BENCH8_PY=${BENCH8_PY:-$HOME/Apps/vllm/venv-7.14.0_0.32.0/bin/python}
BENCH8_SEQS=${BENCH8_SEQS:-8}
BENCH8_CTX=${BENCH8_CTX:-32768}
MTP_PROBE=${MTP_PROBE:-$HERE/mtp_probe.py}
VISION_PROBE=${VISION_PROBE:-$HERE/vision_probe.py}

EVAL_TREE=${EVAL_TREE:-$HOME/w4a8_runs/inkling-eval/tree}
EVAL_PY=${EVAL_PY:-python3}
EVAL_WORKERS=${EVAL_WORKERS:-4}

KL_MIXED=${KL_MIXED:-$HOME/w4a8_runs/inkling-hi2/stage4/calib/heldout_mixed_ids.npy}
KL_LONG_DIR=${KL_LONG_DIR:-$HOME/w4a8_runs/inkling-hi2/stage4/calib_v2}
KL_REF_MIXED=${KL_REF_MIXED:-$HOME/w4a8_runs/inkling-hi2/stage4/eval/bpw3.5_mixed.json}
KL_MIXED_WINDOWS=${KL_MIXED_WINDOWS:-32}
KL_LONG_WINDOWS=${KL_LONG_WINDOWS:-8}

CTX=${CTX:-32768}
CTX_DSPARK=${CTX_DSPARK:-16384}
GMEM_PAIR=${GMEM_PAIR:-0.45}
GMEM_SINGLE=${GMEM_SINGLE:-0.92}
MAXBAT=${MAXBAT:-2048}
SEQS=${SEQS:-4}

MIN_FREE_GIB=${MIN_FREE_GIB:-30}
DETACH=${DETACH:-1}
SETSID=${SETSID-setsid}
RESUME=${RESUME:-1}
DRY_RUN=${DRY_RUN:-0}
ONLY=${ONLY:-}
TAG=${TAG:-$(date -u +%Y%m%dT%H%M%SZ)}
RUN=${RUN:-$RT/final/$TAG}
BOOT_WAIT=${BOOT_WAIT:-1800}
QUIET_WAIT=${QUIET_WAIT:-1800}

STEPS=(kernel serve_dir smoke kl eval vision bench bench8 boot720 dspark)
IFS=, read -r -a _only <<<"$ONLY"
want() {
    [[ -z ${ONLY//,/} ]] && return 0
    local s
    for s in "${_only[@]}"; do [[ $s == "$1" ]] && return 0; done
    return 1
}

mkdir -p "$RUN"/{steps,logs,pids} || exit 3
STATUS=$RUN/STATUS.log
log() {
    local line
    line="[$(date -u +%H:%M:%S)] $*"
    printf '%s\n' "$line"
    printf '%s\n' "$line" >>"$STATUS"
}
die() {
    log "FAIL preflight: $*"
    exit 3
}

# --- guards ----------------------------------------------------------------
check_disk() {
    local free_kib need_kib
    free_kib=$(df -Pk "$RT" | awk 'NR==2 {print $4}')
    need_kib=$((MIN_FREE_GIB * 1024 * 1024))
    [[ -n $free_kib ]] || die "cannot read free space of $RT"
    if ((free_kib < need_kib)); then
        die "disk guard: $((free_kib / 1024 / 1024)) GiB free under $RT, need >= $MIN_FREE_GIB GiB"
    fi
    log "disk guard: $((free_kib / 1024 / 1024)) GiB free under $RT (>= $MIN_FREE_GIB GiB)"
}

# One lease for the whole run: the kernel step above took and released its own
# window; every boot from here inherits this holder (LEASE_INHERIT=1) instead of
# taking a nested request, which would deadlock behind itself.
lease_begin() {
    export SLOT OWNER LEASE_DIR EXPECT_PCI_BUSES HIP_VISIBLE_DEVICES
    # shellcheck source=tools/inkling_hi2/lease.sh
    . "$HERE/lease.sh"
    lease_acquire || die "outer lease acquire failed"
    trap 'stop_all; lease_release' EXIT
    trap 'stop_all; lease_release; exit 130' TERM INT
    LEASE_INHERIT=1
    log "outer lease held for the run: $OWNER.$SLOT (boots inherit it)"
}

# --- per-step bookkeeping ---------------------------------------------------
STEP_NUMS=()
num() { STEP_NUMS+=("$1=$2"); }

write_json() {
    local name=$1 rc=$2 secs=$3 logfile=$4
    "$V" - "$RUN/steps/$name.json" "$name" "$rc" "$secs" "$logfile" "${STEP_NUMS[@]}" <<'PY'
import json
import sys

out, name, rc, secs, logfile, *pairs = sys.argv[1:]
numbers = {}
for pair in pairs:
    key, _, value = pair.partition("=")
    numbers[key] = value
with open(out, "w") as f:
    json.dump(
        {
            "step": name,
            "rc": int(rc),
            "seconds": int(secs),
            "numbers": numbers,
            "log": logfile,
        },
        f,
        indent=2,
    )
    f.write("\n")
PY
}

summary() {
    local verdict=$1
    "$V" - "$RUN" "$verdict" "$MODEL" "$TAG" "${STEPS[@]}" <<'PY'
import json
import sys
from pathlib import Path

run, verdict, model, tag, *steps = sys.argv[1:]
run = Path(run)
rows = []
for name in steps:
    path = run / "steps" / f"{name}.json"
    if not path.is_file():
        rows.append((name, "NOT RUN", "-", ""))
        continue
    data = json.loads(path.read_text())
    numbers = ", ".join(f"{k}={v}" for k, v in data["numbers"].items())
    rows.append(
        (name, "PASS" if data["rc"] == 0 else "FAIL", data["seconds"], numbers)
    )
lines = [
    f"# HI2 final validation: {verdict}",
    "",
    f"- model: `{model}`",
    f"- tag: `{tag}`",
    f"- results: `{run}`",
    f"- a2 estimate: ~4 h 20 wall (5 h budget); see the driver header",
    "",
    "| step | status | seconds | numbers |",
    "| --- | --- | --- | --- |",
]
for name, status, seconds, numbers in rows:
    lines.append(f"| {name} | {status} | {seconds} | {numbers} |")
lines += [
    "",
    "Boots were grouped: smoke/kl/eval/vision share the stock+HI2 pair (32768",
    "context, GMEM 0.45 each) and the bench step's pair arm uses it too; the",
    "bench solo arm then boots HI2 alone (GMEM 0.92); dspark uses 16384;",
    "boot720 is a single HI2 boot at 720000.  Every boot goes through",
    "serve.sh (lease 28 GiB gate, quiet windows, boot.lock, HIP->PCI",
    "litmus).  Per-step logs and JSON are next to this file.",
    "",
]
(run / "summary.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
PY
}

stop_all() {
    local pidfile pid
    for pidfile in "$RUN"/pids/*.pid; do
        [[ -e $pidfile ]] || continue
        pid=$(cat "$pidfile" 2>/dev/null)
        [[ -n $pid ]] && kill "$pid" 2>/dev/null && log "stopped $(basename "$pidfile") (pid $pid)"
        rm -f "$pidfile"
    done
    sleep 15
}

run_step() {
    local name=$1 fn=$2 rc secs t0 logfile
    [[ $(type -t "$fn") == function ]] || die "step function $fn is missing"
    if ! want "$name"; then
        log "SKIP $name (not in ONLY=$ONLY)"
        return 0
    fi
    if [[ $RESUME == 1 && -f $RUN/steps/$name.pass ]]; then
        log "SKIP $name (already PASS in $RUN)"
        return 0
    fi
    STEP_NUMS=()
    logfile=$RUN/steps/$name.log
    t0=$(date +%s)
    log "BEGIN $name"
    "$fn" >"$logfile" 2>&1
    rc=$?
    secs=$(( $(date +%s) - t0 ))
    write_json "$name" "$rc" "$secs" "$logfile"
    if ((rc != 0)); then
        log "FAIL $name (${secs}s; log $logfile)"
        stop_all
        summary "FAIL"
        exit 1
    fi
    : >"$RUN/steps/$name.pass"
    log "PASS $name (${secs}s)"
}

# --- probes/health ----------------------------------------------------------
health() { curl -sf "http://127.0.0.1:$1/health" >/dev/null 2>&1; }

# The speculator must be on/off as the recipe intends.  vLLM prints the resolved
# config in its "with config:" line; a lost DSPARK/SPEC_CONFIG shows up as
# speculative_config=None, which used to surface only as 0 drafts in the probe.
spec_config_line() { grep -E "speculative_config=" "$1" 2>/dev/null | tail -1; }

require_spec_config() { # logfile on|off what
    local logfile=$1 want=$2 what=$3 line
    line=$(spec_config_line "$logfile")
    if [[ $want == on ]]; then
        if [[ -z $line || $line == *"speculative_config=None"* ]]; then
            log "spec_config_missing: $what booted without the speculator (${line:-no config line in $logfile})"
            return 1
        fi
        if ! grep -q "Inkling-2 DSpark weights" "$logfile"; then
            log "spec_config_missing: $what has no 'Inkling-2 DSpark weights' draft-load line (${line})"
            return 1
        fi
    else
        if [[ $line != *"speculative_config=None"* ]]; then
            log "spec_config_unexpected: $what must boot plain (${line:-no config line in $logfile})"
            return 1
        fi
    fi
    log "spec_config ok: $what ${line}"
}

# The exact environment/argv of one boot; used by boot_server and the dry-run.
boot_env() { # name port model maxlen maxbat dspark vision gmem
    local name=$1 port=$2 model=$3 maxlen=$4 maxbat=$5 dspark=$6 vision=$7 gmem=$8
    printf '%s\n' \
        "SLOT=$SLOT" "HIP_VISIBLE_DEVICES=$HIP_VISIBLE_DEVICES" \
        "EXPECT_PCI_BUSES=$EXPECT_PCI_BUSES" "VENV=$VENV" "MODEL=$model" \
        "SERVER_PID_FILE=$RUN/pids/$name.pid" "PORT=$port" "MAXLEN=$maxlen" \
        "MAXBAT=$maxbat" "SEQS=$SEQS" "GMEM=$gmem" "DSPARK=$dspark" \
        "VISION=$vision" "WAIT_QUIET=1" "LEASE_INHERIT=1"
}

dry_serve() { # label name port model maxlen maxbat dspark vision gmem
    local label=$1
    shift
    local -a envs
    mapfile -t envs < <(boot_env "$@")
    log "dry-run: serve.sh $label"
    env -u ROCR_VISIBLE_DEVICES -u CUDA_VISIBLE_DEVICES PRINT=1 \
        "${envs[@]}" bash "$SERVE" >>"$RUN/dryrun.log" 2>&1 ||
        die "dry-run of the $label serve.sh argv failed (see $RUN/dryrun.log)"
}

boot_server() { # name port model maxlen maxbat dspark vision gmem
    local name=$1 port=$2
    local -a envs
    mapfile -t envs < <(boot_env "$@")
    if health "$port"; then
        if [[ -f $RUN/pids/$name.pid ]] && kill -0 "$(cat "$RUN/pids/$name.pid")" 2>/dev/null; then
            return 0
        fi
        die "port $port answers /health but $name.pid is not ours; refusing to touch a foreign server"
    fi
    log "boot $name on port $port"
    local -a launch=()
    [[ -n ${SETSID-setsid} ]] && launch=("${SETSID-setsid}")
    env -u ROCR_VISIBLE_DEVICES -u CUDA_VISIBLE_DEVICES "${envs[@]}" "${launch[@]}" nohup \
        bash "$SERVE" >"$RUN/logs/serve-$name.log" 2>&1 < /dev/null &
    local boot_pid=$! i
    for ((i = 0; i < BOOT_WAIT / 10; i++)); do
        health "$port" && return 0
        if [[ -f $RUN/pids/$name.pid ]] && ! kill -0 "$(cat "$RUN/pids/$name.pid")" 2>/dev/null; then
            log "boot $name: its pid file is stale; see $RUN/logs/serve-$name.log"
            return 1
        fi
        if ! kill -0 "$boot_pid" 2>/dev/null; then
            log "boot $name: serve.sh exited before /health; see $RUN/logs/serve-$name.log"
            return 1
        fi
        sleep 10
    done
    return 1
}

stop_server() {
    local name=$1 port=$2 pid
    local pidfile=$RUN/pids/$name.pid
    [[ -f $pidfile ]] || return 0
    pid=$(cat "$pidfile" 2>/dev/null)
    [[ -n $pid ]] && kill "$pid" 2>/dev/null && log "stop $name (pid $pid)"
    local i
    for ((i = 0; i < 24; i++)); do
        health "$port" || break
        sleep 5
    done
    rm -f "$pidfile"
}

pair_health() { health 18231 && health 18230; }

# Boot one fresh run pair (stock 18231 + HI2 18230) with the driver's own
# pidfiles; every leftover of ours on those ports is stopped first.
restart_pair() {
    stop_server stock 18231
    stop_server hi2 18230
    stop_server smoke_stock 18231
    stop_server smoke-hi2 18230
    boot_server stock 18231 "$STOCK" "$CTX" "$MAXBAT" 0 1 "$GMEM_PAIR" || return 1
    boot_server hi2 18230 "$SERVE_DIR" "$CTX" "$MAXBAT" 0 1 "$GMEM_PAIR" || return 1
    pair_health
}

# Every step that talks to the pair starts here: if a port is dead, restart the
# pair once; if it still is not healthy, fail the step as pair-unhealthy before
# any Python client runs (no traceback in the step log).
require_pair() { # step name
    local step=$1
    if pair_health; then
        return 0
    fi
    log "pair unhealthy before $step (18231/18230 down); restarting the pair once"
    if restart_pair; then
        log "pair restarted for $step"
        return 0
    fi
    log "pair-unhealthy: $step cannot run; the pair did not come back after one restart (see $RUN/logs)"
    return 1
}

# --- preflight: CPU checks and the dry-run of every GPU argv ----------------
preflight() {
    check_disk
    [[ -x $V ]] || die "V=$V is not executable"
    [[ -d $TREE/tests/models/inkling2_loop ]] || die "TREE=$TREE is not the fork"
    [[ -f $STOCK/model.safetensors.index.json ]] || die "STOCK=$STOCK is not a model dir"
    [[ -f $EVAL_TREE/tools/inkling_eval/cli.py ]] || die "EVAL_TREE=$EVAL_TREE lacks the eval CLI"
    : >"$RUN/dryrun.log"

    log "dry-run: kernel test collection"
    env -u ROCR_VISIBLE_DEVICES -u CUDA_VISIBLE_DEVICES HIP_VISIBLE_DEVICES= \
        PYTHONPATH="$QUANTIZER_TREE:$TREE:$DEPS" "$V" -m pytest \
        "$TREE/tests/models/inkling2_loop/test_hi2_kernel_gpu.py" \
        --collect-only -q >>"$RUN/dryrun.log" 2>&1 ||
        die "kernel test collection failed (see $RUN/dryrun.log)"

    # The serve-dir argv is identical to the stock one (the model path is the only
    # difference and serve_rdna.sh has no per-format flag), so the stock dir is
    # used here: step 2 owns the HI2 directory and runs after the kernel test.
    dry_serve "stock" stock 18231 "$STOCK" "$CTX" "$MAXBAT" 0 1 "$GMEM_PAIR"
    dry_serve "hi2 plain" hi2 18230 "$STOCK" "$CTX" "$MAXBAT" 0 1 "$GMEM_PAIR"
    dry_serve "hi2 dspark" dspark 18230 "$SERVE_DIR" "$CTX_DSPARK" "$MAXBAT" 1 0 "$GMEM_PAIR"
    dry_serve "hi2 plain-16k" plain16k 18231 "$SERVE_DIR" "$CTX_DSPARK" "$MAXBAT" 0 0 "$GMEM_PAIR"
    dry_serve "stock dspark" stockspec 18231 "$STOCK" "$CTX_DSPARK" "$MAXBAT" 1 0 "$GMEM_PAIR"
    dry_serve "hi2 720k" boot720 18230 "$SERVE_DIR" 720000 1024 1 0 "$GMEM_SINGLE"

    local tool
    for tool in "$KLPROBE" "$BENCH" "$MTP_PROBE" "$VISION_PROBE"; do
        log "dry-run: $(basename "$tool") --help"
        "$V" "$tool" --help >>"$RUN/dryrun.log" 2>&1 ||
            die "dry-run of $tool failed (see $RUN/dryrun.log)"
    done
    log "dry-run: eval CLI"
    (cd "$EVAL_TREE" && env -u ROCR_VISIBLE_DEVICES -u CUDA_VISIBLE_DEVICES HIP_VISIBLE_DEVICES= \
        "$EVAL_PY" -m tools.inkling_eval.cli run --help) >>"$RUN/dryrun.log" 2>&1 ||
        die "dry-run of the eval CLI failed (see $RUN/dryrun.log)"
    log "dry-run: bench8 CLI and result schema (find_spec, no vllm import)"
    # The bench8 client imports vllm, which needs the GPU runtime; on CPU this
    # locates the package without executing it and asserts the flags/keys the
    # step relies on exist in that CLI's serve.py.  The argv is echoed too.
    if ! "$BENCH8_PY" - >>"$RUN/dryrun.log" 2>&1 <<'PY'
import importlib.util
import pathlib

spec = importlib.util.find_spec("vllm")
root = next(iter(spec.submodule_search_locations))
serve = pathlib.Path(root) / "benchmarks" / "serve.py"
text = serve.read_text()
needed = (
    "save_result", "result_filename", "base_url", "ignore_eos",
    "percentiles_ttft_ms", "percentiles_itl_ms", "completed",
    "output_throughput",
)
missing = [name for name in needed if name not in text]
if missing:
    raise SystemExit(f"bench8 CLI {serve} lacks {missing}")
print("bench8 CLI ok:", serve)
PY
    then
        die "dry-run of the bench8 CLI failed (see $RUN/dryrun.log)"
    fi
    b8_argv stock 1024 512 16 '<served-id>' "$STOCK"
    {
        echo "bench8 cell 1 (stock): env -u ROCR_VISIBLE_DEVICES -u CUDA_VISIBLE_DEVICES HIP_VISIBLE_DEVICES=0,1 \"$BENCH8_PY\" ${B8_ARGS[*]}"
        b8_argv stock 16384 512 8 '<served-id>' "$STOCK"
        echo "bench8 cell 2 (stock): env -u ROCR_VISIBLE_DEVICES -u CUDA_VISIBLE_DEVICES HIP_VISIBLE_DEVICES=0,1 \"$BENCH8_PY\" ${B8_ARGS[*]}"
        b8_argv hi2 1024 512 16 '<served-id>' "$SERVE_DIR"
        echo "bench8 cell 3 (hi2): env -u ROCR_VISIBLE_DEVICES -u CUDA_VISIBLE_DEVICES HIP_VISIBLE_DEVICES=0,1 \"$BENCH8_PY\" ${B8_ARGS[*]}"
        b8_argv hi2 16384 512 8 '<served-id>' "$SERVE_DIR"
        echo "bench8 cell 4 (hi2): env -u ROCR_VISIBLE_DEVICES -u CUDA_VISIBLE_DEVICES HIP_VISIBLE_DEVICES=0,1 \"$BENCH8_PY\" ${B8_ARGS[*]}"
    } >>"$RUN/dryrun.log"
    log "preflight: every GPU argv dry-ran on CPU; no lease taken yet"
}

# --- step 1: kernel test (own lease window) --------------------------------
s_kernel() {
    if [[ ! -f $TINY/hi2/model.safetensors.index.json ]]; then
        log "build tiny HI2 fixture under $TINY"
        env -u ROCR_VISIBLE_DEVICES -u CUDA_VISIBLE_DEVICES HIP_VISIBLE_DEVICES= \
            PYTHONPATH="$QUANTIZER_TREE:$DEPS:$TREE" \
            INKLING2_TOOLS="$TREE/tools/rdna/inkling2" \
            "$V" "$TREE/tools/inkling_hi2/make_tiny_hi2.py" --out "$TINY" ||
            return 1
    fi
    (
        set -euo pipefail
        export SLOT OWNER LEASE_DIR QUIET_DIR BOOT_LOCK EXPECT_PCI_BUSES
        export HIP_VISIBLE_DEVICES
        unset ROCR_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES
        export LEASE_INHERIT=0
        unset LEASE_INHERIT_ACTIVE
        # shellcheck source=tools/inkling_hi2/lease.sh
        . "$HERE/lease.sh"
        lease_acquire
        trap 'lease_release' EXIT
        env -u ROCR_VISIBLE_DEVICES -u CUDA_VISIBLE_DEVICES \
            HIP_VISIBLE_DEVICES="$HIP_VISIBLE_DEVICES" \
            "$VENV/bin/python" "$HERE/slot.py" --slot "$SLOT" \
            --hip "$HIP_VISIBLE_DEVICES" --expect-buses "$EXPECT_PCI_BUSES"
        mkdir -p "$(dirname "$BOOT_LOCK")" "$QUIET_DIR"
        exec 9>"$BOOT_LOCK"
        flock -w 900 9
        local deadline=$((SECONDS + QUIET_WAIT)) q
        while q=$(find "$QUIET_DIR" -type f -mmin -6 -print -quit 2>/dev/null) && [[ -n $q ]]; do
            ((SECONDS < deadline)) || {
                echo "kernel: quiet window $q did not close within $QUIET_WAIT s" >&2
                exit 1
            }
            sleep 60
        done
        env -u ROCR_VISIBLE_DEVICES -u CUDA_VISIBLE_DEVICES HIP_VISIBLE_DEVICES="$HIP_VISIBLE_DEVICES" \
            PYTHONPATH="$TREE" "HI2_TINY_DIR=$TINY/hi2" \
            "$V" -m pytest "$TREE/tests/models/inkling2_loop/test_hi2_kernel_gpu.py" -q
        rc=$?
        flock -u 9
        exec 9>&-
        lease_release
        trap - EXIT
        exit "$rc"
    ) || return 1
    local line
    line=$(grep -E "[0-9]+ passed" "$RUN/steps/kernel.log" | tail -1)
    num "pytest" "${line:-unknown}"
}

# --- step 2: serve dir + CPU checks ----------------------------------------
s_serve_dir() {
    log "prep_serve_dir -> $SERVE_DIR"
    env -u ROCR_VISIBLE_DEVICES -u CUDA_VISIBLE_DEVICES HIP_VISIBLE_DEVICES= \
        PYTHONPATH="$TREE" REQUIRE_MTP=1 FORCE=1 \
        MTP_STOCK="${MTP_STOCK:-$STOCK/mtp.safetensors}" \
        bash "$PREP" "$MODEL" "$SERVE_DIR" || return 1
    env -u ROCR_VISIBLE_DEVICES -u CUDA_VISIBLE_DEVICES HIP_VISIBLE_DEVICES= \
        PYTHONPATH="$TREE" "$V" "$VALIDATE" --model "$SERVE_DIR" \
        --require-mtp --json "$RUN/steps/serve_dir.validate.json" || return 1
    env -u ROCR_VISIBLE_DEVICES -u CUDA_VISIBLE_DEVICES HIP_VISIBLE_DEVICES= \
        PYTHONPATH="$TREE" "$V" "$CHECK_REAL" --model "$SERVE_DIR" \
        --stock "$STOCK" --vision --mtp || return 1
    local vision mtp dense
    vision=$(grep -oE "vision tensors bit-equal to stock: [0-9]+/[0-9]+" "$RUN/steps/serve_dir.log" | tail -1)
    mtp=$(grep -oE "mtp.safetensors: [0-9]+ tensors bit-equal to stock: True" "$RUN/steps/serve_dir.log" | tail -1)
    dense=$(grep -oE "dense text tensors bit-equal to stock: [0-9]+" "$RUN/steps/serve_dir.log" | tail -1)
    num "vision" "${vision#*: }"
    num "mtp" "$( [[ -n $mtp ]] && echo bit-equal || echo FAIL )"
    num "dense" "${dense##*: }"
}

# --- step 3: smoke (self-contained: its own stock + its own HI2) -----------
# smoke.sh stops what it started, so it must never share a port with the run
# pair: its stock is booted here as smoke_stock, smoke.sh boots/stops its own
# HI2 as smoke-hi2, and both are stopped before the step returns.  The run pair
# is booted fresh by the next require_pair.
s_smoke() {
    boot_server smoke_stock 18231 "$STOCK" "$CTX" "$MAXBAT" 0 1 "$GMEM_PAIR" || {
        log "smoke stock boot failed (see $RUN/logs/serve-smoke_stock.log)"
        return 1
    }
    local rc=0
    env -u ROCR_VISIBLE_DEVICES -u CUDA_VISIBLE_DEVICES \
        MODEL="$SERVE_DIR" SLOT="$SLOT" HIP_VISIBLE_DEVICES="$HIP_VISIBLE_DEVICES" \
        EXPECT_PCI_BUSES="$EXPECT_PCI_BUSES" VENV="$VENV" \
        OWNER="$OWNER" LEASE_DIR="$LEASE_DIR" QUIET_DIR="$QUIET_DIR" \
        BOOT_LOCK="$BOOT_LOCK" LEASE_INHERIT=1 \
        STOCK_URL=http://127.0.0.1:18231 STOCK_KL=0 \
        SERVER_PID_FILE="$RUN/pids/smoke-hi2.pid" PIDS_DIR="$RUN/pids" \
        GMEM="$GMEM_PAIR" MAXLEN="$CTX" MAXBAT="$MAXBAT" SEQS="$SEQS" WAIT_QUIET=1 \
        bash "$SMOKE" || rc=$?
    grep -q "RESULT measured window PASS" "$RUN/steps/smoke.log" || rc=1
    stop_server smoke_stock 18231
    stop_server smoke-hi2 18230
    ((rc == 0)) || return 1
    num "window" PASS
}

# --- step 4: served KL (mixed + long 8k/16k) -------------------------------
s_kl() {
    require_pair kl || return 1
    local spec ids windows gate ref
    for spec in mixed long8k long16k; do
        case $spec in
            mixed) ids=$KL_MIXED; windows=$KL_MIXED_WINDOWS; gate="--max-kl 0.15"; ref="--reference-eval $KL_REF_MIXED" ;;
            long8k) ids=$KL_LONG_DIR/heldout_long_context__en__8192_ids.npy; windows=$KL_LONG_WINDOWS; gate="--min-top1 0"; ref="" ;;
            long16k) ids=$KL_LONG_DIR/heldout_long_context__en__16384_ids.npy; windows=$KL_LONG_WINDOWS; gate="--min-top1 0"; ref="" ;;
        esac
        [[ -f $ids ]] || { log "missing ids file $ids"; return 1; }
        log "KL $spec: $ids ($windows windows)"
        # shellcheck disable=SC2086
        env -u ROCR_VISIBLE_DEVICES -u CUDA_VISIBLE_DEVICES HIP_VISIBLE_DEVICES=0,1 \
            "$V" "$KLPROBE" --hi2-url http://127.0.0.1:18230 \
            --stock-url http://127.0.0.1:18231 --ids "$ids" --windows "$windows" \
            --topk 100 $gate $ref --out "$RUN/steps/kl-$spec.json" || return 1
        local kl_line
        kl_line=$("$V" -c "import json;d=json.load(open('$RUN/steps/kl-$spec.json'));print('mean_kl=%.5f sem=%.5f top1=%.4f windows=%s' % (d['kl_mean'], d.get('kl_sem', 0.0), d['top1'], d['windows']))") ||
            { log "cannot parse $RUN/steps/kl-$spec.json (keys: kl_mean/top1/windows)"; return 1; }
        num "kl_$spec" "$kl_line"
    done
}

# A resumed run skips kl when it already PASSed; the card reads steps/kl.json,
# so fill its empty numbers from the first attempt's probe JSONs.
kl_backfill() {
    [[ -f $RUN/steps/kl.pass && -f $RUN/steps/kl.json ]] || return 0
    local line
    line=$("$V" - "$RUN/steps" <<'PY'
import json
import sys
from pathlib import Path

steps = Path(sys.argv[1])
out = steps / "kl.json"
data = json.loads(out.read_text())
if any(data.get("numbers", {}).values()):
    sys.exit(0)
numbers = {}
for spec in ("mixed", "long8k", "long16k"):
    path = steps / f"kl-{spec}.json"
    if not path.is_file():
        continue
    d = json.loads(path.read_text())
    numbers[f"kl_{spec}"] = "mean_kl=%.5f sem=%.5f top1=%.4f windows=%s" % (
        d["kl_mean"],
        d.get("kl_sem", 0.0),
        d["top1"],
        d["windows"],
    )
if not numbers:
    sys.exit(0)
data["numbers"] = numbers
out.write_text(json.dumps(data, indent=2) + "\n")
print(", ".join(f"{k}={v}" for k, v in numbers.items()))
PY
)
    [[ -n $line ]] && log "kl.json numbers backfilled from the probe JSONs: $line"
    return 0
}

# --- step 7: bench cells, pair arm then a solo arm -------------------------
# The pair arm is the conservative number (two servers resident, GMEM 0.45
# each); the solo arm is the production shape (one HI2 server, GMEM 0.92) and
# is the number the card quotes.  Both use --decode-tokens 128 so a decode step
# is less contaminated by the prefill of the same request.
bench_extract() { # pair_stock pair_hi2 solo
    "$V" - "$RUN/steps/bench-pair-stock.json" "$RUN/steps/bench-pair-hi2.json" \
        "$RUN/steps/bench-solo.json" <<'PY'
import json
import sys

stock, pair, solo = (json.load(open(path)) for path in sys.argv[1:4])


def cell(run, kind, batch=1):
    for c in run["cells"]:
        if c["cell"] == kind and c.get("batch", 1) == batch:
            return c
    raise SystemExit(f"missing {kind} b={batch} in {run.get('label')}")


p0, p1 = cell(stock, "prefill"), cell(pair, "prefill")
d1, d4 = cell(pair, "decode", 1), cell(pair, "decode", 4)
s0, s1, s4 = cell(solo, "prefill"), cell(solo, "decode", 1), cell(solo, "decode", 4)
print(
    "pair_prefill=%.3f pair_decode_b1=%.3f pair_decode_b4=%.3f "
    "solo_prefill_tps=%.1f solo_ms_b1=%.1f solo_ms_b4=%.1f "
    "solo_prefill_vs_pair=%.3f solo_decode_b4_vs_pair=%.3f"
    % (
        p1["prompt_tokens_per_s"] / p0["prompt_tokens_per_s"],
        d1["ms_per_step"] / cell(stock, "decode", 1)["ms_per_step"],
        d4["ms_per_step"] / cell(stock, "decode", 4)["ms_per_step"],
        s0["prompt_tokens_per_s"],
        s1["ms_per_step"],
        s4["ms_per_step"],
        s0["prompt_tokens_per_s"] / p1["prompt_tokens_per_s"],
        s4["ms_per_step"] / d4["ms_per_step"],
    )
)
PY
}

s_bench() {
    local decode=${BENCH_DECODE_TOKENS:-128}
    local flags="gmem=$GMEM_PAIR maxlen=$CTX maxbat=$MAXBAT seqs=$SEQS decode_tokens=$decode"
    require_pair bench || return 1
    env -u ROCR_VISIBLE_DEVICES -u CUDA_VISIBLE_DEVICES HIP_VISIBLE_DEVICES=0,1 \
        "$V" "$BENCH" --url http://127.0.0.1:18231 \
        --server-pid "$(cat "$RUN/pids/stock.pid")" \
        --label stock --log "$RUN/logs/serve-stock.log" \
        --decode-tokens "$decode" --out "$RUN/steps/bench-pair-stock.json" || return 1
    env -u ROCR_VISIBLE_DEVICES -u CUDA_VISIBLE_DEVICES HIP_VISIBLE_DEVICES=0,1 \
        "$V" "$BENCH" --url http://127.0.0.1:18230 \
        --server-pid "$(cat "$RUN/pids/hi2.pid")" \
        --label hi2 --log "$RUN/logs/serve-hi2.log" \
        --decode-tokens "$decode" \
        --baseline "$RUN/steps/bench-pair-stock.json" \
        --out "$RUN/steps/bench-pair-hi2.json" || return 1
    num "decode_tokens" "$decode"
    num "pair_flags" "$flags"
    num "pair_stock" "$RUN/steps/bench-pair-stock.json"
    num "pair_hi2" "$RUN/steps/bench-pair-hi2.json"
    log "pair arm done; stopping the pair for the solo arm"

    stop_server stock 18231
    stop_server hi2 18230
    boot_server solo 18230 "$SERVE_DIR" "$CTX" "$MAXBAT" 0 0 "$GMEM_SINGLE" ||
        { log "solo boot failed (see $RUN/logs/serve-solo.log)"; return 1; }
    env -u ROCR_VISIBLE_DEVICES -u CUDA_VISIBLE_DEVICES HIP_VISIBLE_DEVICES=0,1 \
        "$V" "$BENCH" --url http://127.0.0.1:18230 \
        --server-pid "$(cat "$RUN/pids/solo.pid")" \
        --label hi2-solo --log "$RUN/logs/serve-solo.log" \
        --decode-tokens "$decode" --out "$RUN/steps/bench-solo.json" || return 1
    num "solo_flags" "gmem=$GMEM_SINGLE maxlen=$CTX maxbat=$MAXBAT seqs=$SEQS decode_tokens=$decode"
    num "solo_hi2" "$RUN/steps/bench-solo.json"
    num "ratios" "$(bench_extract)"
    stop_server solo 18230
}

# --- step 7b: vllm bench serve over the HTTP API (the card's b8) -----------
# Unlike step 7 (in-process bench.py with its own serve), this is the stock
# `vllm bench serve` client against the OpenAI endpoint: random prompts,
# --ignore-eos, result JSON saved per cell.  BENCH8_PY: see the variable block.
b8_argv() { # label in out n model_id tokenizer_dir -> B8_ARGS
    # The client resolves the tokenizer before it talks to the server; the
    # served name (inkling2-hi2) is not an HF id, so --tokenizer must be the
    # model directory (the stock dir; it holds the tokenizer files).
    B8_ARGS=(-m vllm.entrypoints.cli.main bench serve
        --backend openai --base-url http://127.0.0.1:18230 --endpoint /v1/completions
        --model "$5" --tokenizer "$6" --dataset-name random
        --random-input-len "$2" --random-output-len "$3" --num-prompts "$4"
        --max-concurrency 8 --ignore-eos --save-result
        --result-dir "$RUN/steps" --result-filename "bench8-$1-$2.json")
}

b8_model_id() {
    curl -sf http://127.0.0.1:18230/v1/models |
        "$V" -c 'import json, sys; print(json.load(sys.stdin)["data"][0]["id"])'
}

b8_extract() { # result.json -> one-line summary of ttft/itl p50/p99 + throughput
    "$V" - "$1" <<'PY'
import json
import sys

d = json.load(open(sys.argv[1]))


def pct(values, p):
    for item in values or []:
        if item and int(item[0]) == p:
            return item[1]
    return None


def pick(values, p, direct):
    value = d.get(direct)
    return pct(values, p) if value is None else value


def fmt(value):
    return "n/a" if value is None else "%.2f" % value


p50_ttft = pick(d.get("percentiles_ttft_ms"), 50, "p50_ttft_ms") or d.get("median_ttft_ms")
p99_ttft = pick(d.get("percentiles_ttft_ms"), 99, "p99_ttft_ms")
p50_itl = pick(d.get("percentiles_itl_ms"), 50, "p50_itl_ms") or d.get("median_itl_ms")
p99_itl = pick(d.get("percentiles_itl_ms"), 99, "p99_itl_ms")
print(
    "ttft_p50=%s ttft_p99=%s itl_p50=%s itl_p99=%s out_tok_s=%s ok=%s failed=%s"
    % (
        fmt(p50_ttft), fmt(p99_ttft), fmt(p50_itl), fmt(p99_itl),
        fmt(d.get("output_throughput")), d.get("completed"), d.get("failed", 0),
    )
)
PY
}

b8_kv() { # tag -> recorded KV-cache usage after the cell
    local kv
    kv=$(curl -sf http://127.0.0.1:18230/metrics 2>/dev/null | "$V" -c '
import sys
for line in sys.stdin:
    if line.startswith(("vllm:kv_cache_usage_perc", "vllm:gpu_cache_usage_perc")):
        print(line.split()[-1])
        break
' 2>/dev/null)
    num "kv_$1" "${kv:-n/a}"
}

b8_window() { # name -> health + the FULL_AND_PIECEWISE graph capture must hold
    local name=$1 lines
    health 18230 || { log "bench8: $name server not healthy"; return 1; }
    lines=$(grep -c "FULL_AND_PIECEWISE" "$RUN/logs/serve-$name.log" 2>/dev/null || true)
    num "${name}_full_and_piecewise_lines" "${lines:-0}"
    if ! grep -q "FULL_AND_PIECEWISE" "$RUN/logs/serve-$name.log"; then
        log "bench8: $name serve log has no FULL_AND_PIECEWISE graph capture"
        return 1
    fi
}

b8_cell() { # label in out n
    local label=$1 in=$2 out=$3 n=$4 file="$RUN/steps/bench8-$1-$2.json" model summary tokenizer
    case $label in
        stock) tokenizer=$STOCK ;;
        *) tokenizer=$SERVE_DIR ;;
    esac
    model=$(b8_model_id) || { log "bench8 $label $in: cannot read /v1/models"; return 1; }
    health 18230 || { log "bench8 $label $in: server unhealthy before the cell"; return 1; }
    b8_argv "$label" "$in" "$out" "$n" "$model" "$tokenizer"
    printf '%s\n' "env -u ROCR_VISIBLE_DEVICES -u CUDA_VISIBLE_DEVICES HIP_VISIBLE_DEVICES=0,1 \"$BENCH8_PY\" ${B8_ARGS[*]}" >>"$RUN/steps/bench8-argv.txt"
    log "bench8 $label: in=$in out=$out n=$n conc=8"
    env -u ROCR_VISIBLE_DEVICES -u CUDA_VISIBLE_DEVICES HIP_VISIBLE_DEVICES=0,1 \
        "$BENCH8_PY" "${B8_ARGS[@]}" >"$RUN/logs/bench8-$label-$in.log" 2>&1 ||
        { log "bench8 $label $in: CLI failed (see logs/bench8-$label-$in.log)"; return 1; }
    [[ -f $file ]] || { log "bench8 $label $in: no result JSON at $file"; return 1; }
    summary=$(b8_extract "$file") || { log "bench8 $label $in: cannot parse $file"; return 1; }
    log "bench8 $label $in: $summary"
    num "cell_${label}_${in}" "$summary"
    b8_kv "${label}_${in}"
    health 18230 || { log "bench8 $label $in: server unhealthy after the cell"; return 1; }
    b8_complete "$file" "$n" ||
        { log "bench8 $label $in: completed != num-prompts or failed > 0"; return 1; }
}

b8_complete() { # result.json n -> 0 iff completed == n and failed == 0
    "$V" - "$1" "$2" <<'PY'
import json
import sys

d = json.load(open(sys.argv[1]))
n = int(sys.argv[2])
completed = d.get("completed")
failed = d.get("failed", 0)
if completed != n or failed:
    raise SystemExit(f"completed={completed} failed={failed} want={n}/0")
PY
}

s_bench8() {
    local ctx=$BENCH8_CTX maxbat=$MAXBAT saved_seqs=$SEQS
    SEQS=$BENCH8_SEQS
    : >"$RUN/steps/bench8-argv.txt"
    num "flags" "maxlen=$ctx maxbat=$maxbat seqs=$SEQS conc=8 cells=1024x512x16,16384x512x8 client=$(basename "$BENCH8_PY")"
    num "argv" "$RUN/steps/bench8-argv.txt"
    stop_server stock 18231
    stop_server hi2 18230
    stop_server b8stock 18230
    stop_server b8hi2 18230

    log "bench8: stock boot (maxlen $ctx maxbat $maxbat seqs $SEQS)"
    boot_server b8stock 18230 "$STOCK" "$ctx" "$maxbat" 0 0 "$GMEM_SINGLE" ||
        { log "bench8: stock boot failed (see logs/serve-b8stock.log)"; SEQS=$saved_seqs; return 1; }
    require_spec_config "$RUN/logs/serve-b8stock.log" off b8stock ||
        { SEQS=$saved_seqs; return 1; }
    b8_window b8stock || { SEQS=$saved_seqs; return 1; }
    b8_cell stock 1024 512 16 || { SEQS=$saved_seqs; return 1; }
    b8_cell stock 16384 512 8 || { SEQS=$saved_seqs; return 1; }
    stop_server b8stock 18230

    log "bench8: HI2 boot"
    boot_server b8hi2 18230 "$SERVE_DIR" "$ctx" "$maxbat" 0 0 "$GMEM_SINGLE" ||
        { log "bench8: HI2 boot failed (see logs/serve-b8hi2.log)"; SEQS=$saved_seqs; return 1; }
    require_spec_config "$RUN/logs/serve-b8hi2.log" off b8hi2 ||
        { SEQS=$saved_seqs; return 1; }
    b8_window b8hi2 || { SEQS=$saved_seqs; return 1; }
    b8_cell hi2 1024 512 16 || { SEQS=$saved_seqs; return 1; }
    b8_cell hi2 16384 512 8 || { SEQS=$saved_seqs; return 1; }
    stop_server b8hi2 18230
    SEQS=$saved_seqs
    num "pass" "4 cells completed==num-prompts, failed=0, healthy around each, FULL_AND_PIECEWISE captured"
}

# --- step 6: task eval (main 245 + agentic-probe + vision) -----------------
s_eval() {
    require_pair eval || return 1
    local R=$RUN/eval
    mkdir -p "$R"
    run_one() { # url label set out
        local url=$1 label=$2 set=$3 out=$4
        local args=(run --url "$url" --label "$label" --out "$out" --workers "$EVAL_WORKERS")
        [[ -n $set ]] && args+=(--set "$set")
        log "eval $label set=${set:-main}"
        (cd "$EVAL_TREE" && env -u ROCR_VISIBLE_DEVICES -u CUDA_VISIBLE_DEVICES \
            HIP_VISIBLE_DEVICES=0,1 "$EVAL_PY" -m tools.inkling_eval.cli "${args[@]}")
    }
    local set url label
    for set in "" agentic-probe vision; do
        case $set in "" | main) tag=main ;; *) tag=$set ;; esac
        run_one http://127.0.0.1:18231 "stock-$tag" "$set" "$R/eval-stock-$tag.jsonl" || return 1
        run_one http://127.0.0.1:18231 "stock2-$tag" "$set" "$R/eval-stock2-$tag.jsonl" || return 1
        run_one http://127.0.0.1:18230 "hi2-$tag" "$set" "$R/eval-hi2-$tag.jsonl" || return 1
    done
    for tag in main agentic-probe vision; do
        log "compare $tag"
        (cd "$EVAL_TREE" && env -u ROCR_VISIBLE_DEVICES -u CUDA_VISIBLE_DEVICES \
            HIP_VISIBLE_DEVICES=0,1 "$EVAL_PY" -m tools.inkling_eval.cli compare \
            --ref "$R/eval-stock-$tag.jsonl" --cand "$R/eval-hi2-$tag.jsonl" \
            --floor "$R/eval-stock2-$tag.jsonl" \
            --out "$R/cmp-$tag.json" --md "$R/cmp-$tag.md" --reps 2000) || return 1
        num "cmp_$tag" "$R/cmp-$tag.md"
    done
}

# A probe defect (404/transport) must not discard the eval vision evidence: the
# eval compare over the full 88-item vision set is the fallback verdict.  It must
# be comparable, paired on >= 50 items and non-inferior within the compare's MDD.
vision_eval_gate() {
    "$V" - "$RUN/eval/cmp-vision.json" <<'PY'
import json
import sys

path = sys.argv[1]
try:
    data = json.load(open(path))
    v = data["categories"]["vision"]
    ok = (
        bool(data.get("comparable"))
        and int(v["n"]) >= 50
        and float(v["cand_acc"]) >= float(v["ref_acc"]) - float(v["mdd"])
    )
    print(
        "n=%s ref_acc=%.4f cand_acc=%.4f delta=%.4f mdd=%.3f comparable=%s"
        % (
            v["n"],
            v["ref_acc"],
            v["cand_acc"],
            float(v["cand_acc"]) - float(v["ref_acc"]),
            v["mdd"],
            data.get("comparable"),
        )
    )
except Exception as e:  # any unusable compare fails the verdict
    print(f"vision eval compare unusable: {e}")
    sys.exit(1)
sys.exit(0 if ok else 1)
PY
}

# --- step 7: vision probe (pair, VISION=1) ---------------------------------
s_vision() {
    require_pair vision || return 1
    local images=$RUN/images
    "$V" "$VISION_PROBE" --make-images "$images" || return 1
    local probe_rc=0
    env -u ROCR_VISIBLE_DEVICES -u CUDA_VISIBLE_DEVICES HIP_VISIBLE_DEVICES=0,1 \
        "$V" "$VISION_PROBE" --hi2-url http://127.0.0.1:18230 \
        --stock-url http://127.0.0.1:18231 --images "$images" --concurrency 4 \
        --json "$RUN/steps/vision.json" || probe_rc=$?
    if ((probe_rc == 0)); then
        num "vision_probe" "PASS"
        num "vision" "$RUN/steps/vision.json"
        return 0
    fi
    if [[ -f $RUN/steps/vision.json ]]; then
        # The probe ran to its verdict and a check failed (an image answered
        # empty/garbled or disagreed with stock): a real image failure stays fatal.
        num "vision_probe" "FAIL (check; $RUN/steps/vision.json)"
        log "vision probe failed a check; keeping it fatal (see $RUN/steps/vision.log)"
        return 1
    fi
    num "vision_probe" "FAIL (probe defect; log $RUN/steps/vision.log)"
    local summary
    if summary=$(vision_eval_gate 2>>"$RUN/steps/vision.log"); then
        num "vision_eval" "$summary"
        log "vision probe failed before a verdict; eval vision compare is non-inferior: $summary"
        return 0
    fi
    log "vision probe failed and the eval vision compare is not usable: $summary"
    return 1
}

# --- step 8: DSpark/MTP (HI2 spec + two plain boots), stock reference ------
# The stack is not bitwise reproducible across processes (TunableOp rows, PLE
# state), so the greedy gate is the cross-process floor: the same 8 prompts on
# TWO plain boots of the same directory.  Phase 2 also re-asks the spec server
# (still up, deterministic) and compares spec-vs-plain2 against plain1-vs-plain2.
# Then a stock DSPARK=1 boot gives the reference acceptance on the same prompts.
s_dspark() {
    stop_server stock 18231
    stop_server hi2 18230
    stop_server plain16k 18231
    stop_server stockspec 18231
    boot_server dspark 18230 "$SERVE_DIR" "$CTX_DSPARK" "$MAXBAT" 1 0 "$GMEM_PAIR" ||
        { log "DSpark boot failed (see $RUN/logs/serve-dspark.log)"; return 1; }
    require_spec_config "$RUN/logs/serve-dspark.log" on dspark || return 1
    # plain boot #1: its outputs are the floor for phase 2
    boot_server plain16k 18231 "$SERVE_DIR" "$CTX_DSPARK" "$MAXBAT" 0 0 "$GMEM_PAIR" ||
        { log "plain boot #1 failed (see $RUN/logs/serve-plain16k.log)"; return 1; }
    require_spec_config "$RUN/logs/serve-plain16k.log" off plain16k || return 1
    env -u ROCR_VISIBLE_DEVICES -u CUDA_VISIBLE_DEVICES HIP_VISIBLE_DEVICES=0,1 \
        "$V" "$MTP_PROBE" --spec-url http://127.0.0.1:18230 \
        --plain-url http://127.0.0.1:18231 --log "$RUN/logs/serve-dspark.log" \
        --drop-batch 3 --settle-seconds "${MTP_SETTLE:-10}" --record-only \
        --json "$RUN/steps/dspark-1.json" || return 1
    stop_server plain16k 18231
    # stock DSPARK=1 reference (same 8 prompts), before the gated phase: its
    # numbers must survive a floor/no-draft finding in phase 2
    boot_server stockspec 18231 "$STOCK" "$CTX_DSPARK" "$MAXBAT" 1 0 "$GMEM_PAIR" ||
        { log "stock DSPARK boot failed (see $RUN/logs/serve-stockspec.log)"; return 1; }
    require_spec_config "$RUN/logs/serve-stockspec.log" on stockspec || return 1
    env -u ROCR_VISIBLE_DEVICES -u CUDA_VISIBLE_DEVICES HIP_VISIBLE_DEVICES=0,1 \
        "$V" "$MTP_PROBE" --stock-only --stock-spec-url http://127.0.0.1:18231 \
        --stock-log "$RUN/logs/serve-stockspec.log" \
        --json "$RUN/steps/dspark-stock.json" || return 1
    num "stock_dspark" "$("$V" -c "import json;n=json.load(open('$RUN/steps/dspark-stock.json'))['numbers'];print('accept_len=%s accept_rate=%s drafts=%s' % (n.get('stock_mean_acceptance_length'), n.get('stock_acceptance_rate'), n.get('stock_drafts')))")"
    stop_server stockspec 18231
    # plain boot #2: the floor (plain vs plain) and spec vs plain2
    boot_server plain16k 18231 "$SERVE_DIR" "$CTX_DSPARK" "$MAXBAT" 0 0 "$GMEM_PAIR" ||
        { log "plain boot #2 failed (see $RUN/logs/serve-plain16k.log)"; return 1; }
    require_spec_config "$RUN/logs/serve-plain16k.log" off plain16k || return 1
    env -u ROCR_VISIBLE_DEVICES -u CUDA_VISIBLE_DEVICES HIP_VISIBLE_DEVICES=0,1 \
        "$V" "$MTP_PROBE" --spec-url http://127.0.0.1:18230 \
        --plain-url http://127.0.0.1:18231 --log "$RUN/logs/serve-dspark.log" \
        --floor-json "$RUN/steps/dspark-1.json" \
        --drop-batch 3 --settle-seconds "${MTP_SETTLE:-10}" \
        --json "$RUN/steps/dspark-probe.json"
    local probe_rc=$?
    # record the numbers even when the probe's gate failed: the finding is the
    # card's evidence (the probe writes its JSON before exiting non-zero)
    num "dspark" "$("$V" -c "import json;n=json.load(open('$RUN/steps/dspark-probe.json'))['numbers'];print('accept_len=%s accept_rate=%s speedup_c1=%s drafts_delta_b3=%s floor_max=%s spec_diff_max=%s' % (n.get('mean_acceptance_length'), n.get('acceptance_rate'), n.get('speedup_c1'), n.get('drafts_delta_at_batch'), n.get('floor_diff_max'), n.get('spec_diff_max')))")"
    [[ $probe_rc == 0 ]] || return 1
    stop_server plain16k 18231
    stop_server dspark 18230
}

# --- step 9: 720k production-context boot ----------------------------------
s_boot720() {
    stop_server dspark 18230
    boot_server boot720 18230 "$SERVE_DIR" 720000 1024 1 0 "$GMEM_SINGLE" ||
        { log "720k boot failed (see $RUN/logs/serve-boot720.log)"; return 1; }
    require_spec_config "$RUN/logs/serve-boot720.log" on boot720 || return 1
    local served_name served_len
    read -r served_name served_len < <(curl -sf http://127.0.0.1:18230/v1/models |
        "$V" -c "import json,sys;d=json.load(sys.stdin)['data'][0];print(d['id'], d.get('max_model_len'))") || return 1
    [[ -n $served_name && -n $served_len ]] || return 1
    [[ $served_len == 720000 ]] || {
        log "/v1/models reports max_model_len=$served_len, not 720000"
        return 1
    }
    local payload txt
    payload=$("$V" -c "import json,sys;print(json.dumps({'model': sys.argv[1], 'messages': [{'role': 'user', 'content': 'Say OK.'}], 'max_tokens': 8, 'temperature': 0}))" "$served_name") || return 1
    txt=$(curl -sf http://127.0.0.1:18230/v1/chat/completions -H 'Content-Type: application/json' \
        -d "$payload" |
        "$V" -c "import json,sys;print(json.load(sys.stdin)['choices'][0]['message']['content'][:40])") || return 1
    log "720k greedy: $txt"
    num "max_model_len" 720000
    num "reply" "$txt"
    stop_server boot720 18230
}

# --- run --------------------------------------------------------------------
if [[ ${VF_DETACHED:-0} != 1 && $DETACH == 1 ]]; then
    log "detaching into $RUN"
    VF_DETACHED=1 setsid nohup env VF_DETACHED=1 \
        MODEL="$MODEL" TAG="$TAG" RUN="$RUN" DETACH=0 ONLY="$ONLY" RESUME="$RESUME" \
        bash "$0" >"$RUN/driver.out" 2>&1 < /dev/null &
    echo "validate_final_a2: started pid $! (RUN=$RUN)"
    echo "  tail -f $RUN/STATUS.log"
    echo "  summary: $RUN/summary.md"
    exit 0
fi

trap 'stop_all' EXIT
log "final validation MODEL=$MODEL TAG=$TAG RUN=$RUN"
log "a2 estimate: ~4 h 20 wall (5 h budget); steps: ${STEPS[*]}"

preflight
if [[ $DRY_RUN == 1 ]]; then
    log "DRY-RUN OK: every GPU argv assembled on CPU; no lease taken"
    summary "DRY-RUN"
    exit 0
fi

run_step kernel s_kernel
lease_begin
run_step serve_dir s_serve_dir
run_step smoke s_smoke
kl_backfill
run_step kl s_kl
run_step eval s_eval
run_step vision s_vision
run_step bench s_bench
run_step bench8 s_bench8
run_step boot720 s_boot720
run_step dspark s_dspark

stop_all
lease_release
trap - EXIT
trap - TERM INT
summary "PASS"
log "DONE: $RUN/summary.md"
