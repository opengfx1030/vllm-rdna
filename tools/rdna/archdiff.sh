#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# archdiff.sh: per-arch disassembly diff of the device code two revisions of
# csrc compile to. It serves the contribution contract: for csrc changes,
# compile for gfx90a;gfx942;gfx950;gfx1030;gfx1100;gfx1200, diff the
# disassembly per function, and let only the intended functions differ.
# CPU only: no GPU is touched.
#
#   BASE=origin/rdna_extras HEAD=my-branch bash tools/rdna/archdiff.sh
#   BASE=origin/rdna_extras HEAD=port/skinny EXPECT=LLGemm1_kernel \
#       MAY_DIFFER="gfx1030 gfx1100 gfx1200" bash tools/rdna/archdiff.sh
#
# Run `bash tools/rdna/archdiff.sh --help` for every knob and the exit codes.
#
# Method. hipcc embeds a per-TU CUID (a hash of the source path and of the whole
# command line) in every code object, so a different tree path or flag set
# rewrites PC-relative immediates even in functions whose source did not change.
# BASE and HEAD are therefore compiled in the SAME tree path and build
# directory: the BASE sources are synced into $WORK/tree (rsync -c keeps the
# mtime of unchanged files), the selected objects are built and copied to
# obj-base/, the HEAD sources are synced over the same tree, the same objects
# are rebuilt and copied to obj-head/. Every phase starts from a fresh hipify
# output and deletes the selected objects first, so no stale object is reused.
# tools/rdna/archdiff_report.py selects the translation units (the csrc files
# that differ plus every ROCm TU that includes one) and compares the objects.
set -euo pipefail
START=$SECONDS

SELF_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
HELPER=$SELF_DIR/archdiff_report.py

log() { printf '%s archdiff: %s\n' "$(date +%T)" "$*" >&2; }
die() {
    log "error: $*"
    exit 2
}
is_on() { [[ ${1:-} == 1 || ${1:-} == true || ${1:-} == yes ]]; }

usage() {
    cat <<'EOF'
Usage: BASE=<git rev | tree dir> HEAD=<git rev | tree dir> bash tools/rdna/archdiff.sh

Builds the csrc translation units that differ between BASE and HEAD for every
arch, disassembles the device code and reports per arch which functions differ.
Revs are resolved in the git repository that contains this script (and exported
with `git archive`); an existing directory is used as a source tree as it is.

Environment:
  BASE, HEAD    git rev or source tree directory (required)
  ARCHES        default "gfx90a;gfx942;gfx950;gfx1030;gfx1100;gfx1200"
  TUS           space separated repo-relative csrc files to build, instead of the
                automatic selection (changed files + every ROCm TU including one)
  ALL           1: build every csrc TU of the ROCm targets (exclusive with TUS)
  EXPECT        ERE matched (search) against DEMANGLED function names that may
                differ, e.g. EXPECT='LLGemm1_kernel'. Unset: nothing may differ.
  MAY_DIFFER    arch list that may differ; arches not listed must have no
                differing function at all ("other arches untouched"). It only
                narrows EXPECT: without EXPECT no function may differ anywhere.
  LINE_SHIFTS   comma list of source line deltas allowed in __LINE__ constants
                (default: derived from the unified diff of the changed files)
  WORK          work dir, default $HOME/w4a8_runs/archdiff/<base10>-<head10>:
                tree/ build/ obj-base/ obj-head/ results/ logs/ (needs a marker
                file .archdiff; a non-empty dir without it is refused)
  OUT           default $WORK/results: archdiff.md, archdiff.json, source.diff,
                plan.json, info.txt
  JOBS          compile jobs, default 8, refused above 8
  BUILD_LOCK    flock file taken around every build step, default
                $HOME/w4a8_runs/audit/build.lock when that directory exists;
                BUILD_LOCK= (empty) takes no lock
  VENV, ROCM    default $HOME/Apps/vllm/venv-7.14.0_0.32.0 and /opt/rocm/core-7.14
  PYTHON        interpreter for the helper (needs `regex`); default: python3 when it
                has `regex` (the venv's python takes seconds to start), else
                $VENV/bin/python
  KEEP_BUILD    1: keep build/ tree/ and the staged sources (default 0: removed
                when the run ends with a verdict; kept on an error)
  COMPARE_ONLY  1: skip staging and building, compare the kept obj-base/ obj-head/
                of an earlier run with the same plan (change EXPECT/MAY_DIFFER)
  DRY_RUN       1: configure BASE, print the objects that would be built, stop
  CCACHE        1: use ccache as compiler launcher (default 0: a path-unique
                tree rarely hits and the shared cache is better left alone)
  EXTRA_HIP_FLAGS, CMAKE_BUILD_TYPE (default RelWithDebInfo), MIN_FREE_GB (5),
  TRITON_KERNELS_SRC_DIR (default: an empty stub, nothing here uses it)

Exit status: 0 nothing differs beyond EXPECT / MAY_DIFFER; 1 an unexpected
difference; 2 usage, build or tool error, or an arch that was not run.
EOF
}

case ${1:-} in
-h | --help)
    usage
    exit 0
    ;;
"") ;;
*)
    usage >&2
    die "unexpected argument '$1' (configure through the environment)"
    ;;
esac

# --- configuration -----------------------------------------------------------
BASE=${BASE:-}
HEAD=${HEAD:-}
if [[ -z $BASE || -z $HEAD ]]; then
    usage >&2
    die "BASE and HEAD are required"
fi

ARCHES=${ARCHES:-gfx90a;gfx942;gfx950;gfx1030;gfx1100;gfx1200}
ARCHES=$(printf '%s' "$ARCHES" | tr ', ' ';;' | tr -s ';')
ARCHES=${ARCHES#;}
ARCHES=${ARCHES%;}
[[ -n $ARCHES ]] || die "ARCHES is empty"
IFS=';' read -ra ARCH_LIST <<<"$ARCHES"
for arch in "${ARCH_LIST[@]}"; do
    [[ $arch =~ ^gfx[0-9]{2,4}[a-z]?$ ]] || die "bad arch '$arch' in ARCHES"
done

JOBS=${JOBS:-8}
if ! [[ $JOBS =~ ^[0-9]+$ ]] || ((JOBS < 1)); then
    die "JOBS must be an integer in 1..8, got '$JOBS'"
fi
((JOBS <= 8)) || die "refusing JOBS=$JOBS (> 8): box rule"

TUS=${TUS:-}
ALL=${ALL:-0}
EXPECT=${EXPECT:-}
KEEP_BUILD=${KEEP_BUILD:-0}
COMPARE_ONLY=${COMPARE_ONLY:-0}
DRY_RUN=${DRY_RUN:-0}
CCACHE=${CCACHE:-0}
MIN_FREE_GB=${MIN_FREE_GB:-5}
CMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE:-RelWithDebInfo}
EXTRA_HIP_FLAGS=${EXTRA_HIP_FLAGS:-}
[[ $MIN_FREE_GB =~ ^[0-9]+$ ]] || die "MIN_FREE_GB must be an integer, got '$MIN_FREE_GB'"
if [[ -n $TUS ]] && is_on "$ALL"; then
    die "TUS and ALL are exclusive"
fi

VENV=${VENV:-$HOME/Apps/vllm/venv-7.14.0_0.32.0}
ROCM=${ROCM:-/opt/rocm/core-7.14}
has_regex() { "$1" -c 'import regex' >/dev/null 2>&1; }
if [[ -z ${PYTHON:-} ]]; then
    for candidate in python3 "$VENV/bin/python"; do
        if command -v "$candidate" >/dev/null && has_regex "$candidate"; then
            PYTHON=$candidate
            break
        fi
    done
    [[ -n ${PYTHON:-} ]] || die "no python with the 'regex' package found (set PYTHON= or VENV=)"
else
    command -v "$PYTHON" >/dev/null || die "python interpreter '$PYTHON' not found"
    has_regex "$PYTHON" || die "$PYTHON cannot import 'regex'"
fi
[[ -f $HELPER ]] || die "missing $HELPER"

# Build lock: unset means the box default when it exists, empty means none.
if [[ -n ${BUILD_LOCK+x} ]]; then
    LOCK=$BUILD_LOCK
elif [[ -d $HOME/w4a8_runs/audit ]]; then
    LOCK=$HOME/w4a8_runs/audit/build.lock
else
    LOCK=
fi

# --- BASE and HEAD -----------------------------------------------------------
REPO=$(git -C "$SELF_DIR" rev-parse --show-toplevel 2>/dev/null || true)

resolve_side() {
    local value=$1 dir commit
    if [[ -d $value ]]; then
        dir=$(cd "$value" && pwd -P)
        printf 'dir|%s|dir-%s\n' "$dir" "$(printf '%s' "$dir" | cksum | cut -d' ' -f1)"
        return 0
    fi
    [[ -n $REPO ]] || die "'$value' is not a directory and $SELF_DIR is not inside a git repository"
    commit=$(git -C "$REPO" rev-parse --verify --quiet "$value^{commit}") ||
        die "cannot resolve '$value' to a commit in $REPO (git fetch it, or pass a tree directory)"
    printf 'rev|%s|%s\n' "$commit" "${commit:0:10}"
}

side=$(resolve_side "$BASE") || exit 2
IFS='|' read -r BASE_KIND BASE_SRC BASE_ID <<<"$side"
side=$(resolve_side "$HEAD") || exit 2
IFS='|' read -r HEAD_KIND HEAD_SRC HEAD_ID <<<"$side"
if [[ $BASE_KIND == dir ]]; then BASE_LABEL=$BASE_SRC; else BASE_LABEL="$BASE (${BASE_SRC:0:10})"; fi
if [[ $HEAD_KIND == dir ]]; then HEAD_LABEL=$HEAD_SRC; else HEAD_LABEL="$HEAD (${HEAD_SRC:0:10})"; fi

# --- work directory ----------------------------------------------------------
WORK=${WORK:-$HOME/w4a8_runs/archdiff/$BASE_ID-$HEAD_ID}
if [[ -e $WORK && ! -d $WORK ]]; then die "WORK=$WORK is not a directory"; fi
if [[ -d $WORK && ! -e $WORK/.archdiff && -n $(ls -A "$WORK") ]]; then
    die "WORK=$WORK is not empty and was not made by archdiff.sh (no .archdiff marker)"
fi
mkdir -p "$WORK"
WORK=$(cd "$WORK" && pwd -P)
case $WORK in
/ | "$HOME") die "refusing WORK=$WORK" ;;
*) ;;
esac
touch "$WORK/.archdiff"

TREE=$WORK/tree
BUILD=$WORK/build
OBJ_BASE=$WORK/obj-base
OBJ_HEAD=$WORK/obj-head
LOGS=$WORK/logs
TMP=$WORK/tmp
PLAN=$WORK/plan.json
OUT=${OUT:-$WORK/results}
mkdir -p "$LOGS" "$TMP" "$OUT"
OUT=$(cd "$OUT" && pwd -P)

# A CPU-only build never needs a GPU: hide them all. No /tmp on the box.
unset ROCR_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES
export HIP_VISIBLE_DEVICES=
export TMPDIR=$TMP

PHASES=""
PHASE_NAME=""
PHASE_T0=0
phase_begin() {
    PHASE_NAME=$1
    PHASE_T0=$SECONDS
    log "phase $1: start"
}
phase_end() {
    local dt=$((SECONDS - PHASE_T0))
    PHASES="$PHASES $PHASE_NAME=${dt}s"
    log "phase $PHASE_NAME: ${dt}s"
}

locked() {
    if [[ -n $LOCK ]]; then
        mkdir -p "$(dirname "$LOCK")"
        flock -n "$LOCK" true || log "waiting for the build lock $LOCK"
        flock "$LOCK" "$@"
    else
        "$@"
    fi
}

# --- helper calls ------------------------------------------------------------
# ROCM tells the helper where the llvm-objdump of the compiler that built the
# objects lives; a distribution llvm-objdump on PATH can be an older LLVM.
helper() { ROCM=$ROCM "$PYTHON" "$HELPER" "$@"; }

compare_and_report() {
    local rc=0 disk
    disk=$(du -sk "$WORK" | cut -f1)
    local info=(--info "work=$WORK" --info "jobs=$JOBS" --info "lock=${LOCK:-none}"
        --info "phases=${PHASES# }" --info "disk before cleanup=$((disk / 1024)) MiB")
    local extra=()
    if [[ -n ${MAY_DIFFER+x} ]]; then extra+=(--may-differ "$MAY_DIFFER"); fi
    if [[ -n ${LINE_SHIFTS:-} ]]; then extra+=(--line-shifts "$LINE_SHIFTS"); fi
    mkdir -p "$TMP/compare"
    helper compare --base-dir "$OBJ_BASE" --head-dir "$OBJ_HEAD" --plan "$PLAN" \
        --arches "$ARCHES" --expect "$EXPECT" --out "$OUT" --tmp "$TMP/compare" \
        "${info[@]}" ${extra[@]+"${extra[@]}"} || rc=$?
    COMPARE_RC=$rc
    return 0
}

cleanup_build() {
    if is_on "$KEEP_BUILD"; then
        log "KEEP_BUILD=1: kept $BUILD $TREE"
        return 0
    fi
    rm -rf "$BUILD" "$TREE" "$WORK/src-base" "$WORK/src-head" "$WORK/deps" "$TMP"
    log "removed build/ tree/ staged sources deps/ and tmp/ of $WORK ($(du -sh "$WORK" | cut -f1) left)"
}

finish() {
    local rc=$1
    {
        printf 'base=%s\nhead=%s\narches=%s\nexit=%s\nphases=%s\n' \
            "$BASE_LABEL" "$HEAD_LABEL" "$ARCHES" "$rc" "${PHASES# }"
        printf 'elapsed=%ss\n' "$((SECONDS - START))"
    } >"$OUT/info.txt"
    cp "$PLAN" "$OUT/plan.json" 2>/dev/null || true
    log "report: $OUT/archdiff.md (exit $rc, ${PHASES# })"
    exit "$rc"
}

COMPARE_RC=0

# --- COMPARE_ONLY: re-judge the objects of an earlier run --------------------
if is_on "$COMPARE_ONLY"; then
    [[ -f $PLAN && -d $OBJ_BASE && -d $OBJ_HEAD ]] ||
        die "COMPARE_ONLY needs $PLAN, $OBJ_BASE and $OBJ_HEAD from an earlier run"
    phase_begin compare
    compare_and_report
    phase_end
    finish "$COMPARE_RC"
fi

# --- stage the sources -------------------------------------------------------
stage_rev() { # commit dest
    rm -rf "$2"
    mkdir -p "$2"
    git -C "$REPO" archive --format=tar "$1" | tar -x -C "$2"
}

phase_begin stage
if [[ $BASE_KIND == rev ]]; then
    stage_rev "$BASE_SRC" "$WORK/src-base"
    SRC_BASE=$WORK/src-base
else
    SRC_BASE=$BASE_SRC
fi
if [[ $HEAD_KIND == rev && $BASE_KIND == rev && $HEAD_SRC == "$BASE_SRC" ]]; then
    SRC_HEAD=$SRC_BASE
elif [[ $HEAD_KIND == rev ]]; then
    stage_rev "$HEAD_SRC" "$WORK/src-head"
    SRC_HEAD=$WORK/src-head
else
    SRC_HEAD=$HEAD_SRC
fi
phase_end

PLAN_ARGS=(--base-tree "$SRC_BASE" --head-tree "$SRC_HEAD"
    --base-label "$BASE_LABEL" --head-label "$HEAD_LABEL")
if [[ -n $TUS ]]; then PLAN_ARGS+=(--tus "$TUS"); fi
if is_on "$ALL"; then PLAN_ARGS+=(--all); fi
if [[ -n ${LINE_SHIFTS:-} ]]; then PLAN_ARGS+=(--line-shifts "$LINE_SHIFTS"); fi

rm -rf "$OBJ_BASE" "$OBJ_HEAD"
mkdir -p "$OBJ_BASE" "$OBJ_HEAD"

phase_begin select
status=$(helper plan "${PLAN_ARGS[@]}" --plan-out "$PLAN" --diff-out "$OUT/source.diff") ||
    die "plan failed"
phase_end

if [[ $status == empty ]]; then
    log "no csrc/ or cmake/ change and no TUS/ALL: nothing to build"
    phase_begin compare
    compare_and_report
    phase_end
    cleanup_build
    finish "$COMPARE_RC"
fi

# --- build environment (only needed from here) -------------------------------
export PATH=$VENV/bin:$ROCM/bin:$PATH
missing=""
for tool in cmake ninja rsync tar; do
    command -v "$tool" >/dev/null || missing="$missing $tool"
done
if [[ -n $LOCK ]]; then command -v flock >/dev/null || missing="$missing flock"; fi
[[ -x $ROCM/bin/hipcc ]] || missing="$missing $ROCM/bin/hipcc"
[[ -x $VENV/bin/python ]] || missing="$missing $VENV/bin/python"
[[ -z $missing ]] || die "missing:$missing"
free_kb=$(df -Pk "$WORK" | awk 'NR == 2 {print $4}')
((free_kb >= MIN_FREE_GB * 1048576)) ||
    die "only $((free_kb / 1048576)) GiB free under $WORK (MIN_FREE_GB=$MIN_FREE_GB)"

# One call: the venv's python is slow to start. sys.path starts with '' for -c.
mkdir -p "$TREE"
venv_info=$(cd "$TREE" && "$VENV/bin/python" -c 'import sys, sysconfig; print(sysconfig.get_paths()["purelib"]); print(":".join(sys.path))')
{
    IFS= read -r SITE
    IFS= read -r PYPATH
} <<<"$venv_info"
export VLLM_TARGET_DEVICE=rocm PYTORCH_ROCM_ARCH=$ARCHES
export VLLM_PYTHON_EXECUTABLE=$VENV/bin/python
export CMAKE_BUILD_TYPE
export CMAKE_HIP_COMPILER=$ROCM/bin/hipcc
export ROCM_HOME=$ROCM ROCM_PATH=$ROCM HIP_PATH=$ROCM HIP_ROOT_DIR=$ROCM
export CMAKE_HIP_COMPILER_ROCM_ROOT=$ROCM
export HIP_DEVICE_LIB_PATH=$ROCM/lib/llvm/amdgcn/bitcode
export LD_LIBRARY_PATH="$SITE/_rocm_sdk_libraries/lib:$SITE/_rocm_sdk_core/lib/host-math/lib:$SITE/_rocm_sdk_core/lib/rocm_sysdeps/lib:$SITE/_rocm_sdk_core/lib:$SITE/torch/lib:${LD_LIBRARY_PATH:-}"
if [[ -z ${TRITON_KERNELS_SRC_DIR:-} ]]; then
    # FetchContent needs an existing non-empty source dir; only the install step
    # of the triton_kernels target reads it, and nothing here installs.
    mkdir -p "$WORK/deps/triton_kernels-stub"
    printf 'stub for configure: archdiff.sh never installs triton_kernels\n' \
        >"$WORK/deps/triton_kernels-stub/README"
    export TRITON_KERNELS_SRC_DIR=$WORK/deps/triton_kernels-stub
fi

sync_tree() { # source dest
    mkdir -p "$2"
    rsync -rlc --delete \
        --exclude=/.git --exclude=/.sha --exclude=/docs --exclude=/bench_results \
        --exclude=__pycache__ --exclude='*.pyc' --exclude=.mypy_cache \
        --exclude=.pytest_cache --exclude=.ruff_cache \
        "$1"/ "$2"/
}

configure() {
    local signature
    signature="arches=$ARCHES type=$CMAKE_BUILD_TYPE rocm=$ROCM venv=$VENV jobs=$JOBS"
    signature="$signature ccache=$CCACHE flags=$EXTRA_HIP_FLAGS"
    if [[ -f $BUILD/.archdiff-config && $(cat "$BUILD/.archdiff-config") != "$signature" ]]; then
        log "configuration changed since the kept build: wiping $BUILD"
        rm -rf "$BUILD"
    fi
    if [[ -f $BUILD/build.ninja ]]; then
        log "reusing the configured $BUILD"
        return 0
    fi
    rm -rf "$BUILD"
    mkdir -p "$BUILD"
    local launcher=()
    if is_on "$CCACHE"; then
        launcher=(-DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache
            -DCMAKE_HIP_COMPILER_LAUNCHER=ccache)
    fi
    local flags=()
    if [[ -n $EXTRA_HIP_FLAGS ]]; then flags=("-DCMAKE_HIP_FLAGS=$EXTRA_HIP_FLAGS"); fi
    locked nice -n 19 cmake -S "$TREE" -B "$BUILD" -G Ninja \
        -DCMAKE_BUILD_TYPE="$CMAKE_BUILD_TYPE" -DVLLM_TARGET_DEVICE=rocm \
        -DVLLM_PYTHON_EXECUTABLE="$VENV/bin/python" "-DVLLM_PYTHON_PATH=$PYPATH" \
        -DFETCHCONTENT_BASE_DIR="$BUILD/deps" \
        -DCMAKE_JOB_POOL_COMPILE:STRING=compile "-DCMAKE_JOB_POOLS:STRING=compile=$JOBS" \
        -DROCM_PATH="$ROCM" ${launcher[@]+"${launcher[@]}"} ${flags[@]+"${flags[@]}"} \
        >"$LOGS/configure.log" 2>&1 ||
        {
            tail -n 30 "$LOGS/configure.log" >&2
            die "configure failed, see $LOGS/configure.log"
        }
    printf '%s\n' "$signature" >"$BUILD/.archdiff-config"
}

dump_compdb() { # dest
    locked nice -n 19 ninja -C "$BUILD" build.ninja >>"$LOGS/configure.log" 2>&1 ||
        die "regenerating the build graph failed, see $LOGS/configure.log"
    ninja -C "$BUILD" -t compdb >"$1" || die "ninja -t compdb failed"
}

check_arches() { # compdb
    local arch
    for arch in "${ARCH_LIST[@]}"; do
        grep -q -- "--offload-arch=$arch" "$1" ||
            die "no compile command targets $arch: the tree's CMakeLists drops it (override_gpu_arches)"
    done
}

plan_phase() { # compdb objects-out [previous-plan]
    local prev=()
    if [[ -n ${3:-} ]]; then prev=(--previous "$3"); fi
    helper plan "${PLAN_ARGS[@]}" --compdb "$1" --build-dir "$BUILD" --tree-dir "$TREE" \
        --plan-out "$PLAN" --objects-out "$2" ${prev[@]+"${prev[@]}"} >/dev/null ||
        die "plan failed"
}

build_phase() { # side objects-file dest
    local side=$1 list=$2 dest=$3 obj
    local targets=()
    while IFS= read -r obj; do
        if [[ -n $obj ]]; then targets+=("$obj"); fi
    done <"$list"
    if ((${#targets[@]} == 0)); then
        log "$side: no object to build"
        return 0
    fi
    # Fresh hipify output and no object left over from an earlier phase or run.
    rm -rf "$BUILD/csrc"
    for obj in "${targets[@]}"; do rm -f "$BUILD/$obj"; done
    log "$side: building ${#targets[@]} object(s) for $ARCHES"
    locked nice -n 19 cmake --build "$BUILD" -j "$JOBS" --target "${targets[@]}" \
        >"$LOGS/build-$side.log" 2>&1 ||
        {
            tail -n 40 "$LOGS/build-$side.log" >&2
            die "$side build failed, see $LOGS/build-$side.log"
        }
    for obj in "${targets[@]}"; do
        [[ -f $BUILD/$obj ]] || die "$side build did not produce $obj"
    done
    rm -rf "$dest"
    mkdir -p "$dest"
    rsync -a --files-from="$list" "$BUILD/" "$dest/"
}

phase_begin "sync-base"
sync_tree "$SRC_BASE" "$TREE"
phase_end

phase_begin "configure"
configure
dump_compdb "$TMP/compdb-base.json"
check_arches "$TMP/compdb-base.json"
phase_end

phase_begin "plan-base"
plan_phase "$TMP/compdb-base.json" "$TMP/objects-base.txt"
phase_end

if is_on "$DRY_RUN"; then
    log "DRY_RUN: objects selected on the BASE graph:"
    cat "$TMP/objects-base.txt" >&2
    log "DRY_RUN: stopped before building (work dir $WORK)"
    exit 0
fi

phase_begin "build-base"
build_phase base "$TMP/objects-base.txt" "$OBJ_BASE"
phase_end

phase_begin "sync-head"
sync_tree "$SRC_HEAD" "$TREE"
dump_compdb "$TMP/compdb-head.json"
cp "$PLAN" "$TMP/plan-base.json"
plan_phase "$TMP/compdb-head.json" "$TMP/objects-head.txt" "$TMP/plan-base.json"
phase_end

phase_begin "build-head"
build_phase head "$TMP/objects-head.txt" "$OBJ_HEAD"
phase_end

phase_begin "compare"
compare_and_report
phase_end

cleanup_build
finish "$COMPARE_RC"
