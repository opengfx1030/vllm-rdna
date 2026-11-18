#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Final native qualification campaign for online EPLB.
#
#   MODE=r1 BASE_TREE=/built/base HEAD_TREE=/built/head VENV=/venv \
#     MODEL=/model OUT=/new/run bash tools/rdna/eplb/qualify_final.sh
#   MODE=r2 ... [DRY_RUN=1] bash tools/rdna/eplb/qualify_final.sh
#
# r1: one unchanged tools/rdna/pr_receipt.sh receipt at the final head:
#     MODE=abba FAIL_FAST=1 EXACT=1 STALL=1, the default 8-cell matrix and the
#     default exact cells, RECIPE=flashnext-mtp2, independent base/head trees
#     and extension binaries.  Speed verdicts are reported, never gated.
# r2: four fresh boots of HEAD_TREE in the order A1 off, B1 chaos, B2 chaos,
#     A2 off, all with EXTRA_ARGS=--monokernel; off VLLM_RDNA_EPLB=0, on
#     VLLM_RDNA_EPLB=1 VLLM_RDNA_EPLB_CHAOS=8 VLLM_RDNA_EPLB_DECISION_STEPS=512;
#     cells 1024:512:1 1024:512:8 16384:1024:1 16384:1024:8 with their native
#     request counts; BENCH_EXTRA='--temperature 0', CONC_PROBE=1,
#     STALL_PROBE=0, RESET_PREFIX_CACHE=0, READY_TIMEOUT=2400.  The existing
#     hardened tools.rdna.eplb.itl_gate compares A1+A2 against B1+B2 with its
#     default 0.5 %/off-spread envelope and must return all eight gated rows.
#
# Every actual boot first resolves HIP 2,3,4,5 to the b4 PCI buses, V620 name
# and gfx1030 in a separate torch process with HIP_VISIBLE_DEVICES=2,3,4,5 and
# refuses a mismatch.  A thin wrapper around the canonical
# tools/rdna/port_v031/serve_validate.sh then starts the existing tempwatch on
# the four resolved buses, snapshots the AER counters of the endpoints and
# their ancestor bridges, monitors serve.log for the shared fault phrases and
# stops its own server and validator on a live fault, latching OUT/ABORT so no
# later boot retries.  CPU guard helpers run with HIP_VISIBLE_DEVICES empty and
# never import torch; only the separate GPU resolver does.
#
# Fixed: GPUS=2,3,4,5, RECIPE=flashnext-mtp2, BOOT_LOCK=~/w4a8_runs/boot.lock,
# QUIET_DIR=~/w4a8_runs/lanes/quiet-b4, capped quiet renewal (QUIET_WIN_MAX=240)
# and the existing boot-lock/buildx waiting.  REFUSED_GPUS keeps HIP 0 and 1
# protected; this script never clears it.  Taking and releasing the b4 lease is
# the caller's job.
#
# DRY_RUN=1 checks the inputs and the native launch plan without a GPU access:
# r1 runs pr_receipt.sh DRY_RUN=1 (its PRINT plans), r2 prints the head
# serve_rdna.sh PRINT plan and the four arm command lines.  Use a fresh OUT for
# the real run: the dry run writes manifest/plan files there.
#
# Exit: 0 qualified, 1 not qualified/incomplete, 2 bad input, 3 a boot's
# validator reported an abort (thermal or busy).  Raw boots stay under OUT.
# Arm environments deliberately change OUT/QUALIFY_WRAPPER only in subshells.
# shellcheck disable=SC2030,SC2031
set -uo pipefail

SELF=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")
TOOLS=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
QUALIFY_BOOT=$TOOLS/tools/rdna/eplb/qualify_boot.py
CANONICAL_VALIDATE=$TOOLS/tools/rdna/port_v031/serve_validate.sh
PR_RECEIPT=$TOOLS/tools/rdna/pr_receipt.sh
TEMPWATCH=$TOOLS/tools/rdna/tp8pp2/tempwatch.py
GPUS_FIXED=2,3,4,5
RECIPE_FIXED=flashnext-mtp2
R2_CELLS="1024:512:1 1024:512:8 16384:1024:1 16384:1024:8"
R2_PORT=18248
QUALIFY_PYTHON=${QUALIFY_PYTHON:-/usr/bin/python3}

QUALIFY_CHILD=""
QUALIFY_TW=""
QUALIFY_OUT=""

die() {
    echo "qualify: $*" >&2
    exit 2
}

note() {
    echo "[$(date +%H:%M:%S)] qualify: $*"
}

latch() {
    local path=${QUALIFY_ABORT:-}
    [[ -n $path ]] || path=${OUT:-}/ABORT
    [[ -n ${OUT:-} ]] || return 0
    printf '%s %s\n' "$(date -u '+%F %T')" "$*" >> "$path"
}

# A guard helper: no GPU is visible and torch is never imported here.
guard_run() {
    env -u ROCR_VISIBLE_DEVICES -u CUDA_VISIBLE_DEVICES \
        HIP_VISIBLE_DEVICES= PYTHONPATH="$TOOLS" "$@"
}

write_guard() { # py dir ok note validator-rc faults buses
    local py=$1 dir=$2 ok=$3 text=$4 rc=$5 faults=$6 buses=$7
    guard_run "$py" "$QUALIFY_BOOT" guard --out "$dir/guard.json" --ok "$ok" \
        --buses "$buses" --faults "$faults" --validator-rc "$rc" \
        --note "$text" --abort "$dir/THERMAL_ABORT" > /dev/null 2>&1 || true
}

# Stop this boot's own server only: the pid comes from its serve.pid and its
# command line must still be the launcher or the vLLM worker.  Never a group
# whose identity cannot be checked.
stop_owned() {
    local pidfile=$1/serve.pid pid cmd waited=0 grace=${STOP_GRACE:-120}
    [[ -r $pidfile ]] || return 0
    pid=$(tr -dc '0-9' < "$pidfile")
    [[ -n $pid ]] || return 0
    cmd=$(tr '\0' ' ' < "/proc/$pid/cmdline" 2> /dev/null) || return 0
    [[ -n $cmd ]] || return 0
    case $cmd in
        *vllm* | *serve_rdna.sh*) ;;
        *)
            echo "qualify: not stopping pid $pid (not this boot's server)" >&2
            return 0
            ;;
    esac
    if [[ $(ps -o pgid= -p "$pid" 2> /dev/null | tr -d ' ') != "$pid" ]]; then
        echo "qualify: not stopping pid $pid (not a server process group)" >&2
        return 0
    fi
    if kill -0 -- "-$pid" 2> /dev/null; then
        kill -TERM -- "-$pid" 2> /dev/null
        while ((waited < grace)) && kill -0 -- "-$pid" 2> /dev/null; do
            sleep 2
            waited=$((waited + 2))
        done
        kill -KILL -- "-$pid" 2> /dev/null
    fi
}

# Invoked by the signal trap installed by qualify_serve_validate.
# shellcheck disable=SC2329
qualify_cleanup() {
    [[ -z $QUALIFY_OUT ]] || stop_owned "$QUALIFY_OUT"
    [[ -z $QUALIFY_CHILD ]] || kill -TERM "$QUALIFY_CHILD" 2> /dev/null
    [[ -z $QUALIFY_TW ]] || kill "$QUALIFY_TW" 2> /dev/null
}

# The thin SERVE_VALIDATE wrapper: r1 passes it to pr_receipt.sh and r2 calls
# it directly.  It returns 0 only when the canonical validator exited 0, the
# guard telemetry is complete and unchanged, and no fault was seen.
qualify_serve_validate() {
    local out=${OUT:?OUT required}
    local campaign=${QUALIFY_CAMPAIGN_OUT:?QUALIFY_CAMPAIGN_OUT required}
    local abort=${QUALIFY_ABORT:-$campaign/ABORT}
    local py=$QUALIFY_PYTHON
    local tag=${TAG:-boot}
    local buses="" rc=0 why="" tw=0 child=0 frc=0 fault_count=0
    local -a problems=()
    [[ -x $py ]] || {
        echo "qualify: no CPU guard python at $py" >&2
        return 1
    }
    QUALIFY_OUT=$out
    if [[ -e $abort ]]; then
        echo "qualify: campaign abort latched: $(tail -n 1 "$abort")" >&2
        return 3
    fi
    mkdir -p "$out"
    rm -f "$out/serve.pid" "$out/THERMAL_ABORT"
    trap 'qualify_cleanup; exit 1' INT TERM
    echo "qualify: $tag: resolving HIP $GPUS_FIXED to PCI buses in a torch process"
    if ! buses=$(guard_run "$py" "$QUALIFY_BOOT" gpus --venv "$VENV" \
        --out "$out/gpus.json" 2> "$out/gpus.err"); then
        why="GPU mapping refused: $(tail -n 1 "$out/gpus.err")"
        latch "$tag: $why"
        write_guard "$py" "$out" 0 "$why" 1 0 ""
        echo "qualify: $tag REFUSED: $why" >&2
        trap - INT TERM
        return 1
    fi
    case $buses in
        "d9 dc df e2") ;;
        *)
            why="GPU mapping printed '$buses', expected 'd9 dc df e2'"
            latch "$tag: $why"
            write_guard "$py" "$out" 0 "$why" 1 0 "$buses"
            echo "qualify: $tag REFUSED: $why" >&2
            trap - INT TERM
            return 1
            ;;
    esac
    echo "qualify: $tag: buses=$buses"
    if ! guard_run "$py" "$QUALIFY_BOOT" aer --root /sys --buses "$buses" \
        --out "$out/aer-before.json" > "$out/aer-before.log" 2>&1; then
        why="AER snapshot before the boot failed: $(tail -n 1 "$out/aer-before.log")"
        latch "$tag: $why"
        write_guard "$py" "$out" 0 "$why" 1 0 "$buses"
        echo "qualify: $tag REFUSED: $why" >&2
        trap - INT TERM
        return 1
    fi
    echo "qualify: $tag: starting the temperature watcher on the slot buses"
    env -u ROCR_VISIBLE_DEVICES -u CUDA_VISIBLE_DEVICES HIP_VISIBLE_DEVICES= \
        "$py" "$TEMPWATCH" "${buses// /,}" "$out/temps-gpus.log" \
        "$out/temps-all.log" "$out/serve.pid" "$out/THERMAL_ABORT" \
        2> "$out/tempwatch.err" &
    tw=$!
    QUALIFY_TW=$tw
    sleep 2
    if ! kill -0 "$tw" 2> /dev/null; then
        if [[ -e $out/THERMAL_ABORT ]]; then
            why="thermal abort: $(tr '\n' ' ' < "$out/THERMAL_ABORT")"
        else
            why="the temperature watcher did not run: $(head -c 300 "$out/tempwatch.err")"
        fi
        latch "$tag: $why"
        write_guard "$py" "$out" 0 "$why" 1 1 "$buses"
        echo "qualify: $tag REFUSED: $why" >&2
        QUALIFY_TW=""
        trap - INT TERM
        return 1
    fi
    echo "qualify: $tag: booting the canonical validator"
    HIP_VISIBLE_DEVICES="$GPUS_FIXED" PYTHONPATH="$TOOLS" \
        bash "$CANONICAL_VALIDATE" "$@" > "$out/validator.log" 2>&1 &
    child=$!
    QUALIFY_CHILD=$child
    while kill -0 "$child" 2> /dev/null; do
        sleep "${QUALIFY_POLL:-5}"
        if [[ -e $out/THERMAL_ABORT ]]; then
            why="thermal abort: $(tr '\n' ' ' < "$out/THERMAL_ABORT")"
            break
        fi
        if [[ -e $abort ]]; then
            why="campaign abort latched: $(tail -n 1 "$abort")"
            break
        fi
        if ! kill -0 "$tw" 2> /dev/null; then
            why="the temperature watcher exited: $(head -c 300 "$out/tempwatch.err")"
            break
        fi
        if ! guard_run "$py" "$QUALIFY_BOOT" aer --root /sys --buses "$buses" \
            --out "$out/aer-live.json" > "$out/aer-live.log" 2>&1; then
            why="live AER telemetry failed: $(tail -n 1 "$out/aer-live.log")"
            break
        elif ! guard_run "$py" "$QUALIFY_BOOT" aer-compare \
            "$out/aer-before.json" "$out/aer-live.json" \
            > "$out/aer-live-check.log" 2>&1; then
            why="live AER counters changed: $(tail -n 1 "$out/aer-live-check.log")"
            break
        fi
        guard_run "$py" "$QUALIFY_BOOT" faults --log "$out/serve.log" \
            > "$out/faults.live" 2> /dev/null
        frc=$?
        if ((frc == 1)); then
            why="serve.log fault: $(head -n 3 "$out/faults.live" | tr '\n' ' ')"
            break
        fi
        if ((frc == 3)) && [[ -e $out/serve.pid || -e $out/serve.log ]]; then
            why="serve.log became unavailable after launch"
            break
        fi
        if ((frc != 0 && frc != 3)); then
            why="serve.log fault scan failed (rc=$frc)"
            break
        fi
    done
    if [[ -n $why ]]; then
        latch "$tag: $why"
        stop_owned "$out"
        kill -TERM "$child" 2> /dev/null
        wait "$child" 2> /dev/null
        QUALIFY_CHILD=""
        kill "$tw" 2> /dev/null
        wait "$tw" 2> /dev/null
        QUALIFY_TW=""
        guard_run "$py" "$QUALIFY_BOOT" aer --root /sys --buses "$buses" \
            --out "$out/aer-after.json" > "$out/aer-after.log" 2>&1 || true
        write_guard "$py" "$out" 0 "$why" 1 1 "$buses"
        echo "qualify: $tag STOPPED: $why" >&2
        trap - INT TERM
        return 1
    fi
    wait "$child"
    rc=$?
    QUALIFY_CHILD=""
    kill "$tw" 2> /dev/null
    wait "$tw" 2> /dev/null
    QUALIFY_TW=""
    stop_owned "$out"
    ((rc == 0)) || problems+=("canonical validator rc=$rc")
    if ! guard_run "$py" "$QUALIFY_BOOT" aer --root /sys --buses "$buses" \
        --out "$out/aer-after.json" > "$out/aer-after.log" 2>&1; then
        problems+=("AER snapshot after the boot failed: $(tail -n 1 "$out/aer-after.log")")
    elif ! guard_run "$py" "$QUALIFY_BOOT" aer-compare \
        "$out/aer-before.json" "$out/aer-after.json" \
        > "$out/aer-compare.log" 2>&1; then
        problems+=("AER counters: $(tail -n 1 "$out/aer-compare.log")")
    fi
    if ! guard_run "$py" "$QUALIFY_BOOT" temps --log "$out/temps-gpus.log" \
        --buses "$buses" --abort "$out/THERMAL_ABORT" \
        > "$out/temps-check.log" 2>&1; then
        problems+=("temperature telemetry: $(tail -n 1 "$out/temps-check.log")")
    fi
    if guard_run "$py" "$QUALIFY_BOOT" faults --log "$out/serve.log" \
        > "$out/faults.log" 2>&1; then
        fault_count=0
    else
        frc=$?
        fault_count=$frc
        problems+=("serve.log fault scan rc=$frc: $(head -n 1 "$out/faults.log")")
    fi
    if ((${#problems[@]})); then
        latch "$tag: ${problems[0]}"
        write_guard "$py" "$out" 0 "$(printf '%s; ' "${problems[@]}")" "$rc" \
            "$fault_count" "$buses"
        echo "qualify: $tag NOT VALID" >&2
        printf '  - %s\n' "${problems[@]}" >&2
        trap - INT TERM
        return 1
    fi
    write_guard "$py" "$out" 1 "ok" 0 0 "$buses"
    echo "qualify: $tag VALID (buses $buses, validator rc 0)"
    trap - INT TERM
    return 0
}

tree_sha() {
    local sha=""
    if [[ -f $1/.sha ]]; then
        sha=$(tr -d '[:space:]' < "$1/.sha")
    fi
    if [[ -z $sha ]]; then
        sha=$(git -C "$1" rev-parse HEAD 2> /dev/null)
    fi
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
    local file
    file=$(find "$1/vllm" -maxdepth 1 -name '_rocm_C*.so' -print -quit 2> /dev/null)
    if [[ -z $file ]]; then
        echo none
        return 0
    fi
    file_md5 "$file"
}

unset_vllm_eplb() {
    local name
    while IFS= read -r name; do
        [[ -n $name ]] && unset "$name"
    done < <(compgen -e | grep '^VLLM_RDNA_EPLB' || true)
    unset VLLM_RDNA2_MOE_DETERMINISTIC
}

require_tree() { # label path
    local label=$1 path=$2
    [[ -d $path ]] || die "$label=$path is not a directory"
    [[ -f $path/tools/rdna/serve_rdna.sh ]] \
        || die "$label=$path has no tools/rdna/serve_rdna.sh"
    [[ -f $path/tools/rdna/recipes/$RECIPE_FIXED.env ]] \
        || die "$label=$path has no recipe $RECIPE_FIXED"
    [[ $(rocm_c_md5 "$path") != none ]] \
        || die "$label=$path has no built vllm/_rocm_C*.so"
}

preflight() {
    : "${BASE_TREE:?BASE_TREE required}" "${HEAD_TREE:?HEAD_TREE required}"
    : "${VENV:?VENV required}" "${MODEL:?MODEL required}" "${OUT:?OUT required}"
    [[ -d ${BASE_TREE:-} ]] || die "BASE_TREE is not a directory"
    [[ -d ${HEAD_TREE:-} ]] || die "HEAD_TREE is not a directory"
    BASE_TREE=$(cd "$BASE_TREE" && pwd)
    HEAD_TREE=$(cd "$HEAD_TREE" && pwd)
    [[ $BASE_TREE != "$HEAD_TREE" ]] || die "BASE_TREE and HEAD_TREE are the same"
    [[ -x $VENV/bin/python ]] || die "VENV=$VENV has no bin/python"
    [[ -d $VENV/lib/python3.12/site-packages ]] \
        || die "VENV=$VENV has no python3.12 site-packages"
    [[ -d $MODEL ]] || die "MODEL=$MODEL is not a directory"
    [[ -f $MODEL/config.json ]] || die "MODEL=$MODEL has no config.json"
    [[ -f $TEMPWATCH ]] || die "$TEMPWATCH is missing"
    [[ -f $QUALIFY_BOOT ]] || die "$QUALIFY_BOOT is missing"
    [[ -f $PR_RECEIPT ]] || die "$PR_RECEIPT is missing"
    [[ -f $CANONICAL_VALIDATE ]] || die "$CANONICAL_VALIDATE is missing"
    if [[ -n ${GPUS:-} && $GPUS != "$GPUS_FIXED" ]]; then
        die "GPUS is fixed to $GPUS_FIXED, not $GPUS"
    fi
    GPUS=$GPUS_FIXED
    if [[ -n ${RECIPE:-} && $RECIPE != "$RECIPE_FIXED" ]]; then
        die "RECIPE is fixed to $RECIPE_FIXED, not $RECIPE"
    fi
    RECIPE=$RECIPE_FIXED
    REFUSED_GPUS=${REFUSED_GPUS-0,1}
    local gpu
    for gpu in 0 1; do
        case ",${REFUSED_GPUS// /}," in
            *",$gpu,"*) ;;
            *) die "REFUSED_GPUS=$REFUSED_GPUS would allow HIP $gpu" ;;
        esac
    done
    require_tree BASE_TREE "$BASE_TREE"
    require_tree HEAD_TREE "$HEAD_TREE"
    [[ -f $HEAD_TREE/tools/rdna/eplb/itl_gate.py ]] \
        || die "HEAD_TREE=$HEAD_TREE has no tools/rdna/eplb/itl_gate.py"
    BASE_SHA=$(tree_sha "$BASE_TREE")
    HEAD_SHA=$(tree_sha "$HEAD_TREE")
    TOOLS_SHA=$(tree_sha "$TOOLS")
    [[ $BASE_SHA != unknown ]] || die "BASE_TREE has no .sha and no git HEAD"
    [[ $HEAD_SHA != unknown ]] || die "HEAD_TREE has no .sha and no git HEAD"
    [[ $TOOLS_SHA != unknown ]] || die "$TOOLS has no .sha and no git HEAD"
    [[ $HEAD_SHA == "$TOOLS_SHA" ]] \
        || die "HEAD_TREE is $HEAD_SHA but the campaign tools are $TOOLS_SHA"
    [[ $BASE_SHA != "$HEAD_SHA" ]] || die "base and head name the same commit"
    BASE_ROCM=$(rocm_c_md5 "$BASE_TREE")
    HEAD_ROCM=$(rocm_c_md5 "$HEAD_TREE")
    [[ $BASE_ROCM != "$HEAD_ROCM" ]] \
        || die "base and head share the $BASE_ROCM extension binary, not independent"
    QUIET_DIR=${QUIET_DIR:-$HOME/w4a8_runs/lanes/quiet-b4}
    BOOT_LOCK=${BOOT_LOCK:-$HOME/w4a8_runs/boot.lock}
    export GPUS RECIPE REFUSED_GPUS QUIET_DIR BOOT_LOCK
    if ((DRY_RUN)); then
        export HIP_VISIBLE_DEVICES=
    else
        export HIP_VISIBLE_DEVICES=$GPUS
    fi
    unset ROCR_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES PCI_BUSES
    if [[ -e $OUT ]]; then
        [[ -d $OUT ]] || die "OUT=$OUT is not a directory"
        [[ -z $(find "$OUT" -mindepth 1 -maxdepth 1 -print -quit) ]] \
            || die "OUT=$OUT is not empty"
    else
        mkdir -p "$OUT" || die "cannot create OUT=$OUT"
    fi
    OUT=$(cd "$OUT" && pwd)
    export OUT
    if ((DRY_RUN == 0)); then
        [[ -d ${HOME:?}/w4a8_runs ]] || die "$HOME/w4a8_runs is missing"
        [[ -d $QUIET_DIR ]] || die "QUIET_DIR=$QUIET_DIR is missing"
        [[ -d $(dirname "$BOOT_LOCK") ]] \
            || die "BOOT_LOCK=$BOOT_LOCK has no parent directory"
    fi
    {
        echo "mode=$MODE"
        echo "gpus=$GPUS"
        echo "recipe=$RECIPE"
        echo "base_tree=$BASE_TREE"
        echo "base_sha=$BASE_SHA"
        echo "head_tree=$HEAD_TREE"
        echo "head_sha=$HEAD_SHA"
        echo "tools_sha=$TOOLS_SHA"
        echo "base_rocm_c=$BASE_ROCM"
        echo "head_rocm_c=$HEAD_ROCM"
        echo "boot_lock=$BOOT_LOCK"
        echo "quiet_dir=$QUIET_DIR"
    } > "$OUT/plan.txt"
    note "mode=$MODE gpus=$GPUS recipe=$RECIPE out=$OUT"
    note "base=$BASE_TREE ($BASE_SHA) head=$HEAD_TREE ($HEAD_SHA)"
    note "boot lock=$BOOT_LOCK quiet=$QUIET_DIR"
}

r1_campaign() {
    local rc=0
    local -a dry=()
    ((DRY_RUN)) && dry=(DRY_RUN=1 "PYTHON=$QUALIFY_PYTHON")
    note "r1: pr_receipt.sh MODE=abba FAIL_FAST=1 EXACT=1 STALL=1, default cells"
    (
        unset CELLS EXACT_CELLS QUALITY_WINDOWS STALL_DECODERS TEMPWATCH
        unset STALL_REPEATS STALL_LENS EXTRA_ARGS EAGER PYTHON
        exec env \
            BASE_TREE="$BASE_TREE" HEAD_TREE="$HEAD_TREE" VENV="$VENV" \
            BASE_VENV="$VENV" HEAD_VENV="$VENV" \
            RECIPE="$RECIPE" MODEL="$MODEL" GPUS="$GPUS" QUALITY=0 \
            MODE=abba FAIL_FAST=1 EXACT=1 STALL=1 OUT="$OUT" \
            CHANGE_CLASS="opt-in RDNA2 EPLB/Mono; flag-off compatibility" \
            REACHED="gfx1030 V620 TP4 MTP2 F&P; independent base/head extensions" \
            NOT_RUN="Other runtime architectures; stationary EPLB; live K2 injection" \
            TEMPWATCH=0 SERVE_VALIDATE="$SELF" QUALIFY_WRAPPER=1 \
            QUALIFY_CAMPAIGN_OUT="$OUT" QUALIFY_ABORT="$OUT/ABORT" \
            QUALIFY_PYTHON="$QUALIFY_PYTHON" QUALIFY_POLL="${QUALIFY_POLL:-5}" \
            OWNER=EplbQualify BOOT_LOCK="$BOOT_LOCK" QUIET_DIR="$QUIET_DIR" \
            QUIET_WIN_MAX=240 ${dry[@]+"${dry[@]}"} \
            bash "$PR_RECEIPT"
    )
    rc=$?
    if ((rc != 0)); then
        latch "r1: pr_receipt.sh exit $rc"
        note "r1: pr_receipt.sh exit $rc; no retry"
        return "$rc"
    fi
    if ((DRY_RUN)); then
        note "r1 DRY RUN: plan check passed; no GPU, boot or guard ran"
        return 0
    fi
    if ! guard_run "$QUALIFY_PYTHON" "$QUALIFY_BOOT" r1 --run "$OUT" \
        --base-tree "$BASE_TREE" --head-tree "$HEAD_TREE" \
        --base-sha "$BASE_SHA" --head-sha "$HEAD_SHA" \
        --tools-sha "$TOOLS_SHA" --receipt-rc "$rc"; then
        latch "r1: qualification failed"
        note "r1: NOT QUALIFIED (see $OUT/qualification.md)"
        return 1
    fi
    note "r1: QUALIFIED (speed labels reported in $OUT/speed-labels.txt)"
    cat "$OUT/qualification.md"
    return 0
}

r2_plan_check() {
    local plan=$OUT/plans/head.txt rc=0
    mkdir -p "$OUT/plans"
    note "r2 DRY RUN: assembling the head launch plan (no GPU access)"
    (
        cd "$HEAD_TREE" || exit 1
        env -u ROCR_VISIBLE_DEVICES -u CUDA_VISIBLE_DEVICES \
            HIP_VISIBLE_DEVICES="$GPUS" PRINT=1 \
            bash "$HEAD_TREE/tools/rdna/serve_rdna.sh" \
            RECIPE="$RECIPE" MODEL="$MODEL" VENV="$VENV" \
            VLLM_TREE="$HEAD_TREE" PORT="$R2_PORT" HOST=127.0.0.1 \
            EXTRA_ARGS=--monokernel
    ) > "$plan" 2>&1
    rc=$?
    if ((rc != 0)); then
        note "r2 DRY RUN: launch plan failed (see $plan)"
        sed 's/^/    /' "$plan" | tail -20
        return 1
    fi
    cat "$plan"
    {
        echo "cells=$R2_CELLS"
        echo "bench_extra=--temperature 0"
        echo "conc_probe=1 stall_probe=0 reset_prefix_cache=0 ready_timeout=2400"
        echo "quiet_hold=1 quiet_win_max=240 port=$R2_PORT"
        echo "A1 VLLM_RDNA_EPLB=0 EXTRA_ARGS=--monokernel"
        echo "B1 VLLM_RDNA_EPLB=1 VLLM_RDNA_EPLB_CHAOS=8" \
            "VLLM_RDNA_EPLB_DECISION_STEPS=512 EXTRA_ARGS=--monokernel"
        echo "B2 VLLM_RDNA_EPLB=1 VLLM_RDNA_EPLB_CHAOS=8" \
            "VLLM_RDNA_EPLB_DECISION_STEPS=512 EXTRA_ARGS=--monokernel"
        echo "A2 VLLM_RDNA_EPLB=0 EXTRA_ARGS=--monokernel"
        echo "itl_gate: --off OUT/boots/A1 OUT/boots/A2" \
            "--on OUT/boots/B1 OUT/boots/B2 (default 0.5% envelope)"
    } >> "$plan"
    note "r2 DRY RUN: plan check passed; no GPU, boot or guard ran"
    return 0
}

r2_campaign() {
    local campaign=$OUT arm on bootout rc=0 gate_json gate_txt
    mkdir -p "$campaign/boots" "$campaign/logs" \
        "$campaign/cache/triton" "$campaign/cache/inductor"
    for arm in A1 B1 B2 A2; do
        if [[ -e $campaign/ABORT ]]; then
            note "r2: campaign abort latched; not booting $arm"
            return 1
        fi
        on=0
        [[ $arm == B* ]] && on=1
        bootout=$campaign/boots/$arm
        note "r2 arm $arm: VLLM_RDNA_EPLB=$on EXTRA_ARGS=--monokernel"
        (
            unset VLLM_RDNA_EPLB VLLM_RDNA_EPLB_CHAOS \
                VLLM_RDNA_EPLB_DECISION_STEPS VLLM_RDNA_EPLB_STATS_DIR
            export VLLM_RDNA_EPLB=$on
            if ((on)); then
                export VLLM_RDNA_EPLB_CHAOS=8
                export VLLM_RDNA_EPLB_DECISION_STEPS=512
            fi
            export OUT=$bootout TAG="qualify-r2-$arm"
            export SERVE_TREE=$HEAD_TREE VENV=$VENV RECIPE=$RECIPE MODEL=$MODEL
            export HIP_VISIBLE_DEVICES=$GPUS PORT=$R2_PORT
            export CELLS="$R2_CELLS" BENCH_EXTRA='--temperature 0'
            export CONC_PROBE=1 STALL_PROBE=0 RESET_PREFIX_CACHE=0
            export READY_TIMEOUT=2400 EXTRA_ARGS=--monokernel
            export QUIET_HOLD=1 QUIET_OWN=EplbQualify QUIET_WIN_MAX=240
            export BOOT_LOCK=$BOOT_LOCK QUIET_DIR=$QUIET_DIR STOP_GRACE=120
            export TRITON_CACHE_DIR=$campaign/cache/triton
            export TORCHINDUCTOR_CACHE_DIR=$campaign/cache/inductor
            export QUALIFY_WRAPPER=1 QUALIFY_CAMPAIGN_OUT=$campaign
            export QUALIFY_ABORT=$campaign/ABORT
            export QUALIFY_PYTHON
            export QUALIFY_POLL=${QUALIFY_POLL:-5}
            unset ROCR_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES PYTHONPATH EAGER
            unset QUALITY_WINDOWS QUALITY_LIMIT QUALITY_TOP QUALITY_PASSAGES
            unset LOAD_HOOK STALL_ARGS STALL_ITER_LOG
            if ((on)); then
                bash "$SELF" VLLM_RDNA_EPLB=1 VLLM_RDNA_EPLB_CHAOS=8 \
                    VLLM_RDNA_EPLB_DECISION_STEPS=512 EXTRA_ARGS=--monokernel
            else
                bash "$SELF" VLLM_RDNA_EPLB=0 EXTRA_ARGS=--monokernel
            fi
        ) > "$campaign/logs/$arm.log" 2>&1
        rc=$?
        if ((rc != 0)); then
            latch "r2: arm $arm exit $rc"
            note "r2: arm $arm failed (rc=$rc, see logs/$arm.log); no retry"
            return 1
        fi
        if ! guard_run "$QUALIFY_PYTHON" "$QUALIFY_BOOT" boot --dir "$bootout" \
            --arm "$arm" --cells "$R2_CELLS" --head-sha "$HEAD_SHA" \
            --out "$bootout/evidence.json"; then
            latch "r2: arm $arm evidence failed"
            note "r2: arm $arm is not a valid measurement"
            return 1
        fi
    done
    gate_json=$campaign/itl-gate.json
    gate_txt=$campaign/itl-gate.txt
    note "r2: itl_gate off A1 A2 against on B1 B2 (default envelope)"
    (
        cd "$HEAD_TREE" || exit 1
        env -u ROCR_VISIBLE_DEVICES -u CUDA_VISIBLE_DEVICES \
            HIP_VISIBLE_DEVICES= PYTHONPATH="$HEAD_TREE" \
            "$QUALIFY_PYTHON" -m tools.rdna.eplb.itl_gate \
            --off "$campaign/boots/A1" "$campaign/boots/A2" \
            --on "$campaign/boots/B1" "$campaign/boots/B2" \
            --json "$gate_json"
    ) > "$gate_txt" 2>&1
    rc=$?
    cat "$gate_txt"
    if ((rc != 0)); then
        latch "r2: itl_gate exit $rc"
        note "r2: itl_gate exit $rc (1 FAIL, 3 not comparable; no re-roll)"
        return 1
    fi
    if ! guard_run "$QUALIFY_PYTHON" "$QUALIFY_BOOT" gate --json "$gate_json" \
        --cells "$R2_CELLS"; then
        latch "r2: gated rows incomplete"
        note "r2: the eight gated ITL rows are not all pass"
        return 1
    fi
    if ! guard_run "$QUALIFY_PYTHON" "$QUALIFY_BOOT" r2 --run "$campaign" \
        --arms A1 B1 B2 A2 --head-sha "$HEAD_SHA" \
        --gate-json "$gate_json" --cells "$R2_CELLS"; then
        latch "r2: qualification failed"
        note "r2: NOT QUALIFIED (see $campaign/qualification.md)"
        return 1
    fi
    note "r2: QUALIFIED (four arms, eight gated ITL rows)"
    cat "$campaign/qualification.md"
    return 0
}

campaign_main() {
    local mode=${MODE:-} rc=0
    case $mode in
        r1 | r2) ;;
        *) die "MODE=r1|r2 (got '${mode:-}')" ;;
    esac
    DRY_RUN=${DRY_RUN:-0}
    case $DRY_RUN in
        0 | 1) ;;
        *) die "DRY_RUN is 0 or 1, not $DRY_RUN" ;;
    esac
    MODE=$mode
    preflight
    unset_vllm_eplb
    if [[ $mode == r1 ]]; then
        r1_campaign
        rc=$?
    else
        if ((DRY_RUN)); then
            r2_plan_check
            rc=$?
        else
            r2_campaign
            rc=$?
        fi
    fi
    if ((rc != 0)); then
        note "campaign exit $rc; raw results stay in $OUT"
    fi
    exit "$rc"
}

if [[ ${QUALIFY_WRAPPER:-0} == 1 ]]; then
    qualify_serve_validate "$@"
    exit $?
fi

campaign_main "$@"
