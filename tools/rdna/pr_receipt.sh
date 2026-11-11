#!/usr/bin/env bash
# The receipt of one change: a base tree against a head tree on this machine,
# through serve_validate.sh and its probes (.github/CONTRIBUTING.md has the contract).
#
#   BASE_TREE=~/trees/base HEAD_TREE=~/trees/head VENV=~/venv \
#   RECIPE=flashnext-mtp2 MODEL=/path/to/model GPUS=2,3,4,5 \
#   [MODE=ab|abba] [QUALITY=1] \
#     bash tools/rdna/pr_receipt.sh [KEY=value recipe overrides...]
#
# BASE_TREE, HEAD_TREE  built trees (python sources and vllm/_rocm_C*.so of one
#                       commit each; a .sha file or a git checkout names it).
#                       The same path twice is an A/A run.
# VENV                  venv that runs servers and bench clients (BASE_VENV and
#                       HEAD_VENV set them apart). PYTHON overrides every helper.
# RECIPE, MODEL, GPUS   as for serve_validate.sh. Trailing KEY=value arguments
#                       are recipe overrides for both trees; EAGER=1 is refused
#                       (verdicts come from the recipe's production mode).
# MODE=ab               base, head (default); MODE=abba: base, head, head, base on
#                       fresh boots, which is what a speed claim needs.
# EXACT=1               token-exact gate (default); 0 skips it.
# QUALITY=1             adds the quality gate (default 0).
# STALL=1               stall probe on every speed boot (default; 0 skips it): the
#                       hard gate and the relative gate of the contract.
# STALL_REPEATS=3       repeats of each scenario (the gates use the median);
#                       STALL_LENS=4096,16384 prefill lengths; STALL_DECODERS:
#                       decoders (default: the recipe's --max-num-seqs less one, so
#                       that the injected prefill has a slot).
# CELLS                 speed cells "in:out:conc[:prompts] ..." (default: the
#                       contract's matrix, `pr_receipt_report.py cells`); none skips.
# EXACT_CELLS           cells whose greedy texts are hashed on the deterministic
#                       boots (default `pr_receipt_report.py cells --exact`).
# QUALITY_WINDOWS       topk_kl.py windows file; default: QUALITY_LIMIT windows cut
#                       from the tools tree's docs/ with the model's tokenizer.
# QUALITY_LIMIT=64      windows per domain; QUALITY_TOP=20, QUALITY_PASSAGES=6,
#                       MAX_PPL_REL=0.02 (quality_ab.py).
# CHANGE_CLASS, REACHED, NOT_RUN   free text for the receipt (change class of the
#                       table in CONTRIBUTING.md; callers and platforms reached;
#                       platforms not run).
# OUT                   run directory (default ~/w4a8_runs/pr-receipt/<stamp>-<head>
#                       when that tree exists, else ./pr-receipt/...); PORT=18780;
#                       OWNER names the quiet window file (default $USER).
# DRY_RUN=1             check the inputs, print the launch plans and the boots, stop.
# FAIL_FAST=1           stop after a failed or inconclusive token-exact gate.
#
# Boots (a fresh server each, one at a time):
#   exact  base, head  VLLM_RDNA2_MOE_DETERMINISTIC=1: greedy hashes, concurrency probe
#   speed  base, head (abba: head, base) production mode: the cells (with per-request
#                      times), the probes, the stall probe and, with QUALITY=1, the
#                      topk_kl / quality_ab collects of each boot
#   floor  base again (ab with QUALITY=1): the base-vs-base floor of the KL check
# Greedy (--temperature 0), --ignore-eos, seed 12345, prefix cache reset before every
# cell, capture-ladder warm-up after boot, KFD evictions invalidate cells: all
# serve_validate.sh. It takes BOOT_LOCK and waits for other agents' quiet windows;
# the cells are measured inside quiet windows that are renewed (after QUIET_GAP
# seconds without one) once they are QUIET_WIN_MAX (240) seconds old, so that the
# longest cell of the matrix (about 10 minutes) ends inside the 15 minute cap.
# Taking the GPU lease is the caller's job. A GPU holding more than MAX_USED_MIB
# (4096) of VRAM stops the run; TEMPWATCH=1 (default) runs tp8pp2/tempwatch.py.
#
# Before the boots, pr_receipt_scan.py lists the lines the diff adds that look like host
# syncs or unchecked launches (scan.md, scan.json): for the reviewer, never a gate.
#
# Results: $OUT/receipt.md and receipt.json (tables of the PR template), the raw
# boots/<id>/ directories (stall/ holds the stall probe), plans/, quality/. Exit 0:
# every gate that ran passed (token-exact, quality, overlap, hard stall gate, and under
# MODE=abba a regression of the relative stall gate); 1: a gate or a boot failed; 2: bad
# input; 3: aborted (busy GPUs, thermal limit). Speed verdicts are reported, never gated.
set -uo pipefail

TOOLS=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
REPORT=$TOOLS/tools/rdna/pr_receipt_report.py
PORT_DIR=$TOOLS/tools/rdna/port_v031
SERVE_VALIDATE=${SERVE_VALIDATE:-$PORT_DIR/serve_validate.sh}

die() {
    echo "pr_receipt: $*" >&2
    exit 2
}

: "${BASE_TREE:?BASE_TREE required}" "${HEAD_TREE:?HEAD_TREE required}"
: "${RECIPE:?RECIPE required}" "${MODEL:?MODEL required}" "${GPUS:?GPUS required}"
for _t in BASE_TREE HEAD_TREE; do
    [[ -d ${!_t} ]] || die "$_t=${!_t} is not a directory"
    printf -v "$_t" '%s' "$(cd "${!_t}" && pwd)"
done
BASE_VENV=${BASE_VENV:-${VENV:-}}
HEAD_VENV=${HEAD_VENV:-${VENV:-}}
[[ -n $BASE_VENV && -n $HEAD_VENV ]] || die "VENV (or BASE_VENV and HEAD_VENV) required"
pyof() { echo "${PYTHON:-$1/bin/python}"; }
PY=$(command -v "$(pyof "$HEAD_VENV")") || die "no python at $(pyof "$HEAD_VENV")"
MODE=${MODE:-ab}
[[ $MODE == ab || $MODE == abba ]] || die "MODE is ab or abba, not $MODE"
EXACT=${EXACT:-1}
QUALITY=${QUALITY:-0}
QUALITY_LIMIT=${QUALITY_LIMIT:-64}
QUALITY_TOP=${QUALITY_TOP:-20}
QUALITY_PASSAGES=${QUALITY_PASSAGES:-6}
MAX_PPL_REL=${MAX_PPL_REL:-0.02}
FAIL_FAST=${FAIL_FAST:-1}
STALL=${STALL:-1}
STALL_REPEATS=${STALL_REPEATS:-3}
STALL_LENS=${STALL_LENS:-4096,16384}
[[ $STALL == 0 || $STALL == 1 ]] || die "STALL is 0 or 1, not $STALL"
[[ $STALL_REPEATS =~ ^[1-9][0-9]*$ ]] || die "STALL_REPEATS is a positive integer, not $STALL_REPEATS"
[[ $STALL_LENS =~ ^[0-9]+(,[0-9]+)*$ ]] || die "STALL_LENS is a comma list of token counts, not $STALL_LENS"
PORT=${PORT:-18780}
OWNER=${OWNER:-${QUIET_OWN:-${USER:-pr-receipt}}}
OVERRIDES=("$@")
[[ ${EAGER:-0} == 1 ]] && die "EAGER=1: verdicts need the recipe's production mode"
for _o in ${OVERRIDES[@]+"${OVERRIDES[@]}"}; do
    [[ $_o == EAGER=1 ]] && die "EAGER=1: verdicts need the recipe's production mode"
done

rcpt() { "$PY" "$REPORT" "$@"; }
CELLS=${CELLS:-$(rcpt cells)}
CELLS=$(rcpt cells "$CELLS") || die "bad CELLS"
EXACT_CELLS=${EXACT_CELLS:-$(rcpt cells --exact)}
EXACT_CELLS=$(rcpt cells "$EXACT_CELLS") || die "bad EXACT_CELLS"
[[ $EXACT == 1 ]] || EXACT_CELLS=none

tree_sha() {
    local sha=""
    [[ -f $1/.sha ]] && sha=$(tr -d '[:space:]' < "$1/.sha")
    [[ -n $sha ]] || sha=$(git -C "$1" rev-parse HEAD 2> /dev/null)
    echo "${sha:-unknown}"
}
file_md5() {
    if command -v md5sum > /dev/null 2>&1; then
        md5sum "$1" | cut -d' ' -f1
    else
        md5 -q "$1"
    fi
}
rocm_c_md5() {
    local f
    for f in "$1"/vllm/_rocm_C*.so; do
        [[ -f $f ]] && { file_md5 "$f"; return; }
    done
    echo none
}
for _t in BASE HEAD; do
    _tree=${_t}_TREE
    [[ -f ${!_tree}/tools/rdna/serve_rdna.sh ]] || die "$_tree has no tools/rdna/serve_rdna.sh"
    [[ -f ${!_tree}/tools/rdna/recipes/$RECIPE.env ]] || die "$_tree has no recipe $RECIPE"
    printf -v "${_t}_SHA" '%s' "$(tree_sha "${!_tree}")"
    [[ $(rocm_c_md5 "${!_tree}") != none ]] \
        || echo "pr_receipt: warning: $_tree has no built vllm/_rocm_C*.so" >&2
done

STAMP=$(date -u +%Y%m%d-%H%M%S)
if [[ -z ${OUT:-} ]]; then
    if [[ -d $HOME/w4a8_runs ]]; then root=$HOME/w4a8_runs/pr-receipt; else root=$PWD/pr-receipt; fi
    OUT=$root/$STAMP-${HEAD_SHA:0:10}-vs-${BASE_SHA:0:10}
fi
[[ ! -e $OUT/manifest.json ]] || die "$OUT already holds a receipt run"
mkdir -p "$OUT/boots" "$OUT/plans" "$OUT/quality" "$OUT/cache"
OUT=$(cd "$OUT" && pwd)
note() { echo "[$(date +%H:%M:%S)] pr_receipt: $*" | tee -a "$OUT/pr_receipt.log"; }

# Shared-host protocol: the boot lock and the quiet windows exist where the lanes do.
if [[ -d $HOME/w4a8_runs ]]; then
    export BOOT_LOCK=${BOOT_LOCK-$HOME/w4a8_runs/boot.lock}
else
    export QUIET_SKIP=${QUIET_SKIP-1}
fi
export QUIET_OWN=$OWNER
export QUIET_WIN_MAX=${QUIET_WIN_MAX-240}
export HIP_VISIBLE_DEVICES=$GPUS
unset ROCR_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES PYTHONPATH
LIMIT=()
_timeout=$(command -v timeout || command -v gtimeout || true)
[[ -z $_timeout ]] || LIMIT=("$_timeout" --kill-after=120 "${BOOT_TIMEOUT:-10800}")

# The launch plan of each tree (serve_rdna.sh PRINT=1): the receipt shows whether they match.
for _r in base head; do
    if [[ $_r == base ]]; then _tree=$BASE_TREE; _venv=$BASE_VENV; else _tree=$HEAD_TREE; _venv=$HEAD_VENV; fi
    PRINT=1 bash "$_tree/tools/rdna/serve_rdna.sh" RECIPE="$RECIPE" MODEL="$MODEL" VENV="$_venv" \
        VLLM_TREE="$_tree" HIP_VISIBLE_DEVICES="$GPUS" ROCR_VISIBLE_DEVICES= PORT="$PORT" \
        HOST=127.0.0.1 VLLM_CACHE_ROOT="$OUT/cache/$_r/vllm" TRITON_CACHE_DIR="$OUT/cache/$_r/triton" \
        TORCHINDUCTOR_CACHE_DIR="$OUT/cache/$_r/inductor" VLLM_SERVER_DEV_MODE=1 \
        ${OVERRIDES[@]+"${OVERRIDES[@]}"} > "$OUT/plans/$_r.txt" 2>&1 \
        || note "warning: $_tree/tools/rdna/serve_rdna.sh PRINT=1 failed (see plans/$_r.txt)"
done
PLAN_DIFF=$(rcpt plan-diff "$OUT/plans/base.txt" "$OUT/plans/head.txt" \
    --mask "$BASE_TREE=<TREE>" --mask "$HEAD_TREE=<TREE>" \
    --mask "$BASE_VENV=<VENV>" --mask "$HEAD_VENV=<VENV>" \
    --mask "$OUT/cache/base=<CACHE>" --mask "$OUT/cache/head=<CACHE>" \
    --out "$OUT/plans/diff.txt")
COMPILATION=$(rcpt plan-field "$OUT/plans/head.txt" compilation)
if [[ $STALL == 1 && -z ${STALL_DECODERS:-} ]]; then
    _seqs=$(rcpt plan-field "$OUT/plans/head.txt" max-num-seqs)
    if [[ ! $_seqs =~ ^[0-9]+$ ]] || ((_seqs < 2)); then
        die "no --max-num-seqs of at least 2 in plans/head.txt: set STALL_DECODERS"
    fi
    STALL_DECODERS=$((_seqs - 1))
fi
STALL_DECODERS=${STALL_DECODERS:-0}
[[ $STALL_DECODERS =~ ^[0-9]+$ ]] || die "STALL_DECODERS is a number of decoders, not $STALL_DECODERS"
[[ $STALL == 0 ]] || ((STALL_DECODERS >= 1)) || die "STALL_DECODERS is at least 1"

PLAN=()
[[ $EXACT == 1 ]] && PLAN+=(XA:base:exact XB:head:exact)
if [[ $CELLS != none || $QUALITY == 1 || $STALL == 1 ]]; then
    if [[ $MODE == abba ]]; then
        PLAN+=(A1:base:speed B1:head:speed B2:head:speed A2:base:speed)
    else
        PLAN+=(A1:base:speed B1:head:speed)
        [[ $QUALITY == 1 ]] && PLAN+=(A2:base:floor)
    fi
fi
((${#PLAN[@]})) || die "nothing to run: EXACT=0, CELLS=none, STALL=0 and QUALITY=0"

GENERATE_WINDOWS=0
if [[ $QUALITY == 1 && -z ${QUALITY_WINDOWS:-} ]]; then
    QUALITY_WINDOWS=$OUT/quality/windows.json
    GENERATE_WINDOWS=1
fi
[[ $QUALITY == 1 ]] || QUALITY_WINDOWS=""

model_rev() {
    if [[ $MODEL =~ snapshots/([0-9a-f]{40}) ]]; then
        echo "${BASH_REMATCH[1]}"
    else
        git -C "$MODEL" rev-parse HEAD 2> /dev/null || echo "not in the path, not a git checkout"
    fi
}
cfg_sha=n/a
if [[ -f $MODEL/config.json ]]; then
    cfg_sha=$("$PY" -c 'import hashlib,sys; print(hashlib.sha256(open(sys.argv[1],"rb").read()).hexdigest())' "$MODEL/config.json")
fi
shq() { # KEY=value or word, quoted only where the shell needs it
    local k="" v=$1
    if [[ $1 == [A-Za-z_]*=* ]]; then
        k=${1%%=*}=
        v=${1#*=}
    fi
    if [[ $v =~ ^[A-Za-z0-9_./:,+@%-]*$ ]]; then
        printf '%s%s ' "$k" "$v"
    else
        printf "%s'%s' " "$k" "${v//\'/\'\\\'\'}"
    fi
}
REPRO=""
for _w in "BASE_TREE=$BASE_TREE" "HEAD_TREE=$HEAD_TREE" "BASE_VENV=$BASE_VENV" "HEAD_VENV=$HEAD_VENV" \
    "RECIPE=$RECIPE" "MODEL=$MODEL" "GPUS=$GPUS" "MODE=$MODE" "EXACT=$EXACT" "QUALITY=$QUALITY" \
    "STALL=$STALL" "STALL_REPEATS=$STALL_REPEATS" "STALL_LENS=$STALL_LENS" "STALL_DECODERS=$STALL_DECODERS" \
    "CELLS=$CELLS" "EXACT_CELLS=$EXACT_CELLS" bash tools/rdna/pr_receipt.sh \
    ${OVERRIDES[@]+"${OVERRIDES[@]}"}; do
    REPRO+=$(shq "$_w")
done
REPRO=${REPRO% }
rcpt set "$OUT" "mode=$MODE" "recipe=$RECIPE" "model=$MODEL" "model_rev=$(model_rev)" \
    "model_cfg_sha=$cfg_sha" "gpus=$GPUS" "overrides=${OVERRIDES[*]-}" "cells=$CELLS" \
    "exact_cells=$EXACT_CELLS" "exact=$EXACT" "quality=$QUALITY" "quality_limit=$QUALITY_LIMIT" \
    "stall=$STALL" "stall_repeats=$STALL_REPEATS" "stall_lens=$STALL_LENS" "stall_decoders=$STALL_DECODERS" \
    "quality_windows=$QUALITY_WINDOWS" "max_ppl_rel=$MAX_PPL_REL" \
    "base_tree=$BASE_TREE" "base_sha=$BASE_SHA" "base_venv=$BASE_VENV" \
    "base_rocm_c=$(rocm_c_md5 "$BASE_TREE")" \
    "head_tree=$HEAD_TREE" "head_sha=$HEAD_SHA" "head_venv=$HEAD_VENV" \
    "head_rocm_c=$(rocm_c_md5 "$HEAD_TREE")" \
    "tools_tree=$TOOLS" "tools_sha=$(tree_sha "$TOOLS")" "host=$(hostname)" "kernel=$(uname -r)" \
    "date=$(date -u '+%F %T')" "out=$OUT" "plan_diff=$PLAN_DIFF" "compilation=$COMPILATION" \
    "change_class=${CHANGE_CLASS:-}" "reached=${REACHED:-}" "not_run=${NOT_RUN:-}" \
    "command=$REPRO" || die "cannot write $OUT/manifest.json"

"$PY" "$TOOLS/tools/rdna/pr_receipt_scan.py" "$BASE_TREE" "$HEAD_TREE" --json "$OUT/scan.json" \
    --md "$OUT/scan.md" > /dev/null 2> "$OUT/scan.err" \
    || note "warning: the diff scan failed (see scan.err): the receipt says it did not run"
SCAN_NOTE=$(head -n 1 "$OUT/scan.md" 2> /dev/null)
if [[ -n ${DRY_RUN:-} ]]; then
    echo "DRY RUN: $OUT"
    echo "launch plans: $PLAN_DIFF; compilation $COMPILATION"
    echo "boots: ${PLAN[*]}"
    echo "speed cells: $CELLS"
    echo "exactness cells: $EXACT_CELLS"
    echo "quality: $QUALITY (windows ${QUALITY_WINDOWS:-none}, limit $QUALITY_LIMIT)"
    echo "stall probe: $STALL (decoders $STALL_DECODERS, prefill $STALL_LENS, repeats $STALL_REPEATS)"
    echo "diff scan: ${SCAN_NOTE:-did not run}"
    exit 0
fi

# GPUs: the python of each venv names them (PCI bus, gfx target) and the buses
# feed the VRAM and temperature guards.
with_venv_libs() { # venv command...
    local venv=$1 site
    shift
    site=$("$(pyof "$venv")" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])') || return 1
    LD_LIBRARY_PATH=$site/_rocm_sdk_libraries/lib:$site/_rocm_sdk_core/lib/host-math/lib:$site/_rocm_sdk_core/lib/rocm_sysdeps/lib:$site/_rocm_sdk_core/lib:$site/torch/lib "$@"
}
for _r in base head; do
    if [[ $_r == base ]]; then _venv=$BASE_VENV; else _venv=$HEAD_VENV; fi
    with_venv_libs "$_venv" "$(command -v "$(pyof "$_venv")")" "$REPORT" env --gpus "$GPUS" \
        --out "$OUT/env-$_r.json" --buses > "$OUT/env-$_r.txt" 2> "$OUT/env-$_r.err" \
        || die "cannot read the GPUs with $_venv (see $OUT/env-$_r.err)"
done
BUSES=$(cat "$OUT/env-head.txt")
[[ -n $BUSES ]] || echo "pr_receipt: warning: no PCI bus found for GPUS=$GPUS: VRAM and temperature guards are off" >&2

MAX_USED_MIB=${MAX_USED_MIB:-4096}
gpus_idle() {
    local bus dev used
    for bus in $BUSES; do
        for dev in /sys/bus/pci/devices/*:"$bus":00.0; do
            [[ -r $dev/mem_info_vram_used ]] || continue
            used=$(($(< "$dev/mem_info_vram_used") / 1048576))
            if ((used > MAX_USED_MIB)); then
                note "the GPU on bus $bus holds $used MiB of VRAM"
                return 1
            fi
        done
    done
}
port_busy() { (exec 3<> "/dev/tcp/127.0.0.1/$PORT") 2> /dev/null; }
wait_idle() {
    local t0
    t0=$(date +%s)
    while ! gpus_idle || port_busy; do
        if (($(date +%s) - t0 > ${IDLE_WAIT:-600})); then
            note "GPUs or port $PORT still busy after ${IDLE_WAIT:-600} s: stopping"
            exit 3
        fi
        sleep 10
    done
}

TEMPWATCH_PID=""
CHILD=""
FOLLOW=""
stop_helpers() {
    [[ -z $TEMPWATCH_PID ]] || kill "$TEMPWATCH_PID" 2> /dev/null
    [[ -z $FOLLOW ]] || kill "$FOLLOW" 2> /dev/null
    TEMPWATCH_PID=""
    FOLLOW=""
}
REPORTED=0
# shellcheck disable=SC2329  # runs from the EXIT trap
finish() {
    local rc=$?
    trap - EXIT
    if [[ -n $CHILD ]]; then
        kill -TERM "$CHILD" 2> /dev/null
        wait "$CHILD" 2> /dev/null
    fi
    stop_helpers
    ((REPORTED)) || rcpt report "$OUT" > "$OUT/report.txt" 2>&1
    exit "$rc"
}
trap finish EXIT
trap 'note "interrupted"; exit 130' INT TERM

# Quality windows are cut once, from the tools tree, for every boot.
if ((GENERATE_WINDOWS)); then
    note "cutting $QUALITY_LIMIT windows of 2048 tokens from docs/"
    find "$TOOLS/docs" -name '*.md' | sort | xargs cat > "$OUT/quality/docs.txt"
    with_venv_libs "$HEAD_VENV" "$PY" "$PORT_DIR/topk_kl.py" windows --tokenizer "$MODEL" \
        --domain "docs=$OUT/quality/docs.txt@0" --count "$QUALITY_LIMIT" --seq-len 2048 \
        --out "$QUALITY_WINDOWS" > "$OUT/quality/windows.log" 2>&1 \
        || die "cannot cut the quality windows (see $OUT/quality/windows.log)"
fi
if [[ $QUALITY == 1 ]]; then
    [[ -f $QUALITY_WINDOWS ]] || die "QUALITY_WINDOWS=$QUALITY_WINDOWS is not a file"
    rcpt set "$OUT" "quality_windows=$QUALITY_WINDOWS" "quality_windows_sha=$(
        "$PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))["sha256"][:16])' "$QUALITY_WINDOWS"
    )"
fi

run_boot() { # id role kind
    local id=$1 role=$2 kind=$3 tree venv bdir cells bench_extra hold=0 hook="" conc=1 rc t0 py
    local -a det=() stall=()
    if [[ $role == base ]]; then tree=$BASE_TREE; venv=$BASE_VENV; else tree=$HEAD_TREE; venv=$HEAD_VENV; fi
    py=$(command -v "$(pyof "$venv")")
    bdir=$OUT/boots/$id
    mkdir -p "$bdir"
    case $kind in
        exact)
            cells=$EXACT_CELLS
            bench_extra="--temperature 0 --save-detailed"
            det=(VLLM_RDNA2_MOE_DETERMINISTIC=1)
            ;;
        speed)
            cells=$CELLS
            bench_extra="--temperature 0 --save-detailed"
            [[ $cells == none && $STALL != 1 ]] || hold=1
            [[ $QUALITY == 1 ]] && hook=$TOOLS/tools/rdna/pr_receipt_hook.sh
            if [[ $STALL == 1 ]]; then
                stall=(STALL_PROBE=1 STALL_ITER_LOG=0 "STALL_ARGS=--scenarios solo,inject,reverse --decoders $STALL_DECODERS --prefill-lens $STALL_LENS --repeats $STALL_REPEATS --max-starved 0 --min-decode-tokens-during-prefill 1 --gate-stat worst")
            fi
            ;;
        *)
            cells=none
            bench_extra="--temperature 0"
            hook=$TOOLS/tools/rdna/pr_receipt_hook.sh
            conc=0
            ;;
    esac
    wait_idle
    note "boot $id ($role, $kind): $tree"
    t0=$(date -u +%T)
    if [[ -n $BUSES && ${TEMPWATCH:-1} == 1 ]]; then
        "$PY" "$TOOLS/tools/rdna/tp8pp2/tempwatch.py" "${BUSES// /,}" "$bdir/temps-gpus.log" \
            "$bdir/temps-all.log" "$bdir/serve.pid" "$bdir/THERMAL_ABORT" 2> "$bdir/tempwatch.err" &
        TEMPWATCH_PID=$!
        sleep 1
        kill -0 "$TEMPWATCH_PID" 2> /dev/null \
            || note "warning: tempwatch did not start ($(head -c 200 "$bdir/tempwatch.err")): thermal guard off"
    fi
    env RECIPE="$RECIPE" MODEL="$MODEL" GPUS="$GPUS" PORT="$PORT" TAG="prr-$STAMP-$id" \
        OUT="$bdir" CACHE="$OUT/cache/$role/$id" VENV="$venv" SERVE_TREE="$tree" \
        TRITON_CACHE_DIR="$OUT/cache/$role/triton" TORCHINDUCTOR_CACHE_DIR="$OUT/cache/$role/inductor" \
        CELLS="$cells" BENCH_EXTRA="$bench_extra" RESET_PREFIX_CACHE=1 CONC_PROBE="$conc" \
        PREFIX_WORDS="${PREFIX_WORDS:-2600}" ${stall[@]+"${stall[@]}"} \
        QUIET_HOLD="$hold" READY_TIMEOUT="${READY_TIMEOUT:-1800}" STOP_GRACE="${STOP_GRACE:-120}" \
        LOAD_HOOK="$hook" RECEIPT_PY="$py" RECEIPT_TOOLS="$TOOLS" QUALITY_WINDOWS="$QUALITY_WINDOWS" \
        QUALITY_LIMIT="$QUALITY_LIMIT" QUALITY_TOP="$QUALITY_TOP" QUALITY_PASSAGES="$QUALITY_PASSAGES" \
        ${LIMIT[@]+"${LIMIT[@]}"} \
        bash "$SERVE_VALIDATE" VLLM_SERVER_DEV_MODE=1 ${OVERRIDES[@]+"${OVERRIDES[@]}"} \
        ${det[@]+"${det[@]}"} > "$bdir/driver.log" 2>&1 &
    CHILD=$!
    tail -n +1 -f "$bdir/driver.log" &
    FOLLOW=$!
    wait "$CHILD"
    rc=$?
    CHILD=""
    sleep 0.5 # let the log follower print the last lines
    stop_helpers
    if [[ -e $bdir/THERMAL_ABORT ]]; then
        note "thermal abort: $(cat "$bdir/THERMAL_ABORT")"
        rc=3
    fi
    rcpt boot "$OUT" "$id" "$role" "$kind" "boots/$id" "rc=$rc" "tree=$tree" "venv=$venv" \
        "started=$t0" "ended=$(date -u +%T)"
    note "boot $id exited $rc"
    return "$rc"
}

note "run directory $OUT; boots ${PLAN[*]}"
for entry in "${PLAN[@]}"; do
    IFS=: read -r id role kind <<< "$entry"
    run_boot "$id" "$role" "$kind"
    rc=$?
    if ((rc != 0)); then
        note "boot $id failed: stopping"
        exit $((rc == 3 ? 3 : 1))
    fi
    if [[ $id == XB && $FAIL_FAST == 1 ]]; then
        verdict=$(rcpt report "$OUT" 2> /dev/null | sed -n 's/^token-exact: //p')
        note "token-exact gate: $verdict"
        if [[ $verdict == FAIL || $verdict == INCONCLUSIVE ]]; then
            note "stopping (FAIL_FAST=1): more boots would not change the gate"
            exit 1
        fi
    fi
done

if [[ $QUALITY == 1 ]]; then
    B=$OUT/boots
    Q=$OUT/quality
    PAIRS=(--pair "floor=$B/A1/topk.npz,$B/A2/topk.npz")
    if [[ $MODE == abba ]]; then
        PAIRS+=(--pair "head1=$B/A1/topk.npz,$B/B1/topk.npz" --pair "head2=$B/A1/topk.npz,$B/B2/topk.npz")
    else
        PAIRS+=(--pair "head=$B/A1/topk.npz,$B/B1/topk.npz")
    fi
    note "comparing the quality collects"
    "$PY" "$PORT_DIR/topk_kl.py" compare "${PAIRS[@]}" --json "$Q/kl.json" --md "$Q/kl.md" \
        > "$Q/kl.log" 2>&1 || note "topk_kl compare failed (see quality/kl.log)"
    ppl() { # label base-boot head-boot
        "$PY" "$PORT_DIR/quality_ab.py" compare "$B/$2/quality.json" "$B/$3/quality.json" \
            --max-ppl-rel "$MAX_PPL_REL" > "$Q/ppl-$1.txt" 2>&1
    }
    ppl floor A1 A2
    if [[ $MODE == abba ]]; then
        ppl head1 A1 B1
        ppl head2 A1 B2
    else
        ppl head A1 B1
    fi
fi

REPORTED=1
rcpt report "$OUT"
rc=$?
echo "pr_receipt: receipt $OUT/receipt.md (exit $rc)"
exit "$rc"
