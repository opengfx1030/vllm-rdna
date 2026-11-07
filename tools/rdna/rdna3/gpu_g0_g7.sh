#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Steps G0-G8 and G16 of docs/rdna2/rdna3-flashnext-port-2026-11-07.md (section 12) on
# ONE Radeon PRO W7800: HIP 0, PCI bus cb (G8: top-k decode algorithm timing, G16:
# BLAS Tensile libraries; use --to G7 for the original G0-G7), and, only with
# --only G13 or --to G13 (it needs the truncated checkpoint and about 40 minutes), G13:
# the layer-truncated checkpoint end to end. It stops at the first failing step (or test
# group of a step) and prints where the logs are.
#
#   bash tools/rdna/rdna3/gpu_g0_g7.sh --list                   # the steps, no GPU
#   bash tools/rdna/rdna3/gpu_g0_g7.sh --dry-run [--tree DIR]   # commands, no GPU
#   bash tools/rdna/rdna3/gpu_g0_g7.sh [--tree DIR] [--venv DIR] [--owner NAME]
#       [--from G2] [--to G5] [--only G4] [--keep-going]
#
# Rules it enforces (local box rules): every GPU command carries
# HIP_VISIBLE_DEVICES=0 (HIP 5, the other W7800, is never named); the lease file
# <owner>.w7800cb is held in ~/w4a8_runs/lanes/lease for the whole run and the
# run is refused when somebody else holds one; the G7 and G8 measured windows follow
# the quiet-file protocol; the junction temperature of the W7800 is sampled
# every second and the running command is killed at a 5 s mean >= 90 C or any
# sample >= 95 C. The tree is an exact-SHA build of tools/rdna/rdna3/build_fat.sh.
# The venv must carry the gfx1100 device packages (amd-torch-device-gfx1100 and
# rocm-sdk-device-gfx1100): on par1-cs25 only venv-7.14.0 does, the 0.32.0 venvs are
# gfx1030-only and fail the first kernel with hipErrorInvalidImage.
# Logs: ~/w4a8_runs/rdna3_port/gpu/<timestamp>/ (summary.tsv, <label>.log).
# The step_* and helper functions are called indirectly ("step_$s").
# shellcheck disable=SC2329
set -uo pipefail

OWNER=${OWNER:-Rdna3Port}
TREE=${TREE:-$HOME/vllm-rdna3-flashnext}
VENV=${VENV:-$HOME/Apps/vllm/venv-7.14.0}
W7800_BUS=cb
HIP_INDEX=0
LANES=$HOME/w4a8_runs/lanes
LOGROOT=$HOME/w4a8_runs/rdna3_port/gpu
FROM=G0
TO=G16
ONLY=
DRY=0
LIST=0
KEEP_GOING=0
LABELS=

while (($#)); do
    case $1 in
        --tree) TREE=$2; shift 2 ;;
        --venv) VENV=$2; shift 2 ;;
        --owner) OWNER=$2; shift 2 ;;
        --from) FROM=$2; shift 2 ;;
        --to) TO=$2; shift 2 ;;
        --only) ONLY=$2; shift 2 ;;
        --dry-run) DRY=1; shift ;;
        --list) LIST=1; DRY=1; shift ;;
        --keep-going) KEEP_GOING=1; shift ;;
        --labels) LABELS=$2; shift 2 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done

# G13 comes last: it is outside the default range (G0 to G16).
STEPS=(G0 G1 G2 G3 G4 G5 G6 G7 G8 G16 G13)
step_index() {
    local i
    for i in "${!STEPS[@]}"; do
        if [[ ${STEPS[$i]} == "$1" ]]; then echo "$i"; return; fi
    done
    echo "unknown step: $1" >&2
    exit 2
}
for _s in "$FROM" "$TO" ${ONLY:+"$ONLY"}; do step_index "$_s" >/dev/null; done
selected() {
    if [[ -n $ONLY ]]; then [[ $1 == "$ONLY" ]]; return; fi
    (($(step_index "$1") >= $(step_index "$FROM") && $(step_index "$1") <= $(step_index "$TO")))
}

PCI_ADDR=0000:$W7800_BUS:00.0
SITE=$VENV/lib/python3.12/site-packages
LIBS=$SITE/_rocm_sdk_libraries/lib:$SITE/_rocm_sdk_core/lib/host-math/lib
LIBS=$LIBS:$SITE/_rocm_sdk_core/lib/rocm_sysdeps/lib:$SITE/_rocm_sdk_core/lib:$SITE/torch/lib
RUN=$LOGROOT/$(date +%Y%m%d-%H%M%S)
LEASE=$LANES/lease/$OWNER.w7800cb
QUIET=$LANES/quiet/$OWNER
FAILED=0
LAST_QUIET_TOUCH=0
PY=$VENV/bin/python

# Every GPU command is prefixed with GPU_ENV: HIP 0 only, no inherited visibility,
# no family opt-in leaking in. CPU_ENV is the CPU-only equivalent.
GPU_ENV=(env -u ROCR_VISIBLE_DEVICES -u CUDA_VISIBLE_DEVICES -u VLLM_RDNA3_HIP_OPS
    "HIP_VISIBLE_DEVICES=$HIP_INDEX" "PYTHONPATH=$TREE" "LD_LIBRARY_PATH=$LIBS"
    "VLLM_CACHE_ROOT=$LOGROOT/cache" "TRITON_CACHE_DIR=$LOGROOT/cache/triton"
    "PATH=$VENV/bin:$PATH")
CPU_ENV=(env -u ROCR_VISIBLE_DEVICES -u CUDA_VISIBLE_DEVICES -u VLLM_RDNA3_HIP_OPS
    "HIP_VISIBLE_DEVICES=" "PYTHONPATH=$TREE" "LD_LIBRARY_PATH=$LIBS"
    "TMPDIR=$HOME/w4a8_runs/rdna3_port/tmp" "PATH=$VENV/bin:$PATH")

# --- plan data: test groups (paths are relative to the tree) -----------------
Q=tests/kernels/quantization
G2_GROUPS=(
    "$Q/test_gdn_prefill_prep_rdna2.py $Q/test_gdn_prefill_kkt_rdna2.py"
    "$Q/test_gdn_prefill_solve_wy_rdna2.py $Q/test_gdn_prefill_o_rdna2.py"
    "$Q/test_gdn_prefill_delta_h_rdna2.py"
    "$Q/test_gdn_decode_rdna2.py tests/kernels/mamba/test_causal_conv1d_update_rdna2.py tests/kernels/mamba/test_causal_conv1d_null_block.py"
)
G3_GROUPS=(
    "$Q/test_gemv_f16_rdna2.py"
    "tests/kernels/rocm/qwen4_exp/test_qsa_rdna2.py -k not~mrope tests/kernels/rocm/qwen4_exp/test_qsa_rdna2_int4.py tests/kernels/rocm/qwen4_exp/test_qsa_metadata_null_block_rdna2.py"
    "tests/kernels/rocm/qwen4_exp/test_qsa_rdna2.py -k mrope"
    "tests/models/qwen4_exp/test_qsa_metadata_rdna2.py tests/models/qwen4_exp/test_qsa_amd.py"
    "tests/models/qwen4_exp/test_hc_ops_rdna2.py tests/models/qwen4_exp/test_hc_prefill_rdna2.py tests/models/qwen4_exp/test_se_rdna2.py"
)
G4_GROUPS=(
    "$Q/test_rdna3_compile_guards.py $Q/test_rdna3_moe_w4a16.py"
    "tests/kernels/moe/test_rdna2_unquantized_moe.py"
)
# S1b WMMA kernel, staged because it has never run on a GPU: smoke (34 tests), one
# shape per stage (15), then the whole file (227). A failing stage stops the run: do
# not use --keep-going for the first execution of a new kernel.
G4_WMMA=(
    "$Q/test_rdna3_moe_wmma.py -k k128-n16~or~valid_call~or~nothing_routed~or~rejected~or~unsupported"
    "$Q/test_rdna3_moe_wmma.py -k e8-gate_up-m16~or~e8-down-m16"
    "$Q/test_rdna3_moe_wmma.py"
)
# Decode top-k at the serving batches (1, 2 x next_n 1, 8): the batch-2048 cases only
# time the per-row torch.topk reference (~19 s each on this box, 30 min for the file).
G5_TOPK="tests/kernels/test_top_k_per_row.py -k test_top_k_per_row_decode~and~not~2048-2048~and~not~2048-3000"
G5_GROUPS=(
    "$Q/test_rocm_skinny_gemms.py"
    "$G5_TOPK"
)
# Extra variables of the next pytest_group (the top-k algorithm of G8).
PYTEST_EXTRA_ENV=()
G6_GROUPS=(
    "tests/kernels/test_mrope_rdna2.py tests/kernels/core/test_vit_pos_embed_rdna2.py"
    "tests/v1/worker/test_input_batch_ops_rdna2.py tests/v1/worker/test_worker_ops_rdna2.py tests/v1/worker/test_gpu_block_table.py"
    "tests/v1/sample/test_topk_topp_ops_rdna2.py"
)
BENCH=tools/rdna/port_v031
QSA_BENCH=tools/rdna/rdna3/qsa_decode_bench.py
MOE_BENCH=tools/rdna/rdna3/moe_decode_bench.py
TUNE_ENV=PYTORCH_TUNABLEOP_ENABLED=1
# G7 cells: label|script|args|extra env (empty = default route; NAME=value words
# separated by spaces). TP2 = the W7800 pair, TP1 = the single-W7800 hub of the hetero
# stage 1. The shared-expert op does not read the family variable (the model-level gate
# does), so its pair is the fused op against the torch composition the model runs with
# the se family off; the -48 cells raise its row cap (VLLM_RDNA_SE_DECODE_MAX) to the
# 16 sequences x (1 + MTP 2) rows of the hub's widest batch. The lm_head -tune cells
# search the rocBLAS / hipBLASLt solutions (PyTorch TunableOp) and write the table the
# -tuned cells read back; the lm_head-tp* cells above them are the untuned baseline.
G7_CELLS=(
    "hc-default|$BENCH/decode_ops_bench.py|hc --m 3,24,48|"
    "hc-hip|$BENCH/decode_ops_bench.py|hc --m 3,24,48|VLLM_RDNA3_HIP_OPS=hc"
    "se-tp2-torch|$BENCH/decode_ops_bench.py|se --m 3,24,48 --i-se 320 --torch|"
    "se-tp2-op|$BENCH/decode_ops_bench.py|se --m 3,24,48 --i-se 320|"
    "se-tp1-torch|$BENCH/decode_ops_bench.py|se --m 3,24,48 --i-se 640 --torch|"
    "se-tp1-op|$BENCH/decode_ops_bench.py|se --m 3,24,48 --i-se 640|"
    "gdn-tp2-default|tools/rdna/rdna3/gdn_decode_bench.py|--geometry flashnext-tp2 --seqs 1,8,16|"
    "gdn-tp2-hip|tools/rdna/rdna3/gdn_decode_bench.py|--geometry flashnext-tp2 --seqs 1,8,16|VLLM_RDNA3_HIP_OPS=gdn"
    "gdn-tp1-default|tools/rdna/rdna3/gdn_decode_bench.py|--geometry flashnext-tp1 --seqs 1,8,16|"
    "gdn-tp1-hip|tools/rdna/rdna3/gdn_decode_bench.py|--geometry flashnext-tp1 --seqs 1,8,16|VLLM_RDNA3_HIP_OPS=gdn"
    "gdn-tp2-fine-default|tools/rdna/rdna3/gdn_decode_bench.py|--geometry flashnext-tp2 --seqs 2,3,4,6,12|"
    "gdn-tp2-fine-hip|tools/rdna/rdna3/gdn_decode_bench.py|--geometry flashnext-tp2 --seqs 2,3,4,6,12|VLLM_RDNA3_HIP_OPS=gdn"
    "gdn-tp1-fine-default|tools/rdna/rdna3/gdn_decode_bench.py|--geometry flashnext-tp1 --seqs 2,3,4,6,12|"
    "gdn-tp1-fine-hip|tools/rdna/rdna3/gdn_decode_bench.py|--geometry flashnext-tp1 --seqs 2,3,4,6,12|VLLM_RDNA3_HIP_OPS=gdn"
    "lm_head-tp2|$BENCH/lm_head_bench.py|--n 124160 --k 2560 --m 3,24,48|"
    "lm_head-tp1|$BENCH/lm_head_bench.py|--n 248320 --k 2560 --m 3,24,48|"
    "gemv-tp1|$BENCH/gemv_crossover_bench.py|--cold --m 3,24,48 --shapes 16384x2560,96x2560,7168x2560,2560x6144,1280x2560,2560x640,512x2560|"
    "gemv-tp2|$BENCH/gemv_crossover_bench.py|--cold --m 3,24,48 --shapes 8192x2560,48x2560,3584x2560,2560x3072,640x2560,2560x320,512x2560|"
    "se-tp2-op-48|$BENCH/decode_ops_bench.py|se --m 24,32,48 --i-se 320|VLLM_RDNA_SE_DECODE_MAX=48"
    "se-tp1-op-48|$BENCH/decode_ops_bench.py|se --m 24,32,48 --i-se 640|VLLM_RDNA_SE_DECODE_MAX=48"
    "qsa-tp1-default|$QSA_BENCH|--geometry flashnext-tp1 --seqs 1,8,16 --ctx 8192,32768|"
    "qsa-tp1-hip|$QSA_BENCH|--geometry flashnext-tp1 --seqs 1,8,16 --ctx 8192,32768|VLLM_RDNA3_HIP_OPS=qsa"
    "qsa-tp2-default|$QSA_BENCH|--geometry flashnext-tp2 --seqs 1,8,16 --ctx 8192,32768|"
    "qsa-tp2-hip|$QSA_BENCH|--geometry flashnext-tp2 --seqs 1,8,16 --ctx 8192,32768|VLLM_RDNA3_HIP_OPS=qsa"
    "qsa-tp1-hip-capacity|$QSA_BENCH|--geometry flashnext-tp1 --seqs 1,16 --ctx 8192 --capacity 262144 --ops select|VLLM_RDNA3_HIP_OPS=qsa"
    "moe-decode|$MOE_BENCH|--m 1,3,8,24,48|"
    "moe-decode-hub|$MOE_BENCH|--m 1,3,8,24,48 --global-experts 512 --modes default --skip-router|"
    "moe-decode-block-m|$MOE_BENCH|--m 3,8,24,48 --block-m 0,1,2,4,8 --skip-router|"
    "lm_head-tp2-tune|$BENCH/lm_head_bench.py|--n 124160 --k 2560 --m 3,24,48|$TUNE_ENV PYTORCH_TUNABLEOP_TUNING=1 PYTORCH_TUNABLEOP_FILENAME=$RUN/tunableop_lm_head_tp2.csv"
    "lm_head-tp2-tuned|$BENCH/lm_head_bench.py|--n 124160 --k 2560 --m 3,24,48|$TUNE_ENV PYTORCH_TUNABLEOP_TUNING=0 PYTORCH_TUNABLEOP_FILENAME=$RUN/tunableop_lm_head_tp2.csv"
    "lm_head-tp1-tune|$BENCH/lm_head_bench.py|--n 248320 --k 2560 --m 3,24,48|$TUNE_ENV PYTORCH_TUNABLEOP_TUNING=1 PYTORCH_TUNABLEOP_FILENAME=$RUN/tunableop_lm_head_tp1.csv"
    "lm_head-tp1-tuned|$BENCH/lm_head_bench.py|--n 248320 --k 2560 --m 3,24,48|$TUNE_ENV PYTORCH_TUNABLEOP_TUNING=0 PYTORCH_TUNABLEOP_FILENAME=$RUN/tunableop_lm_head_tp1.csv"
)

# G13: the first four layers (3 GDN, 1 QSA) of the real checkpoint, prefill only
# (trunc_checkpoint.py, trunc_e2e.py): per-layer hidden states of base = the W7800 recipe
# as it is (no HIP family), all = every family, wmma = every family plus the S1b MoE
# kernel. The kernel-selecting variables are read from the recipe file itself. Each run
# pins the 95 GiB per-layer-embedding table in host memory (the page cache keeps the file
# after the first), so the step needs that much free RAM. E2E_DIR is where the dumps go:
# point it at an earlier run directory to repeat only the G13-compare-* labels.
TRUNC_CKPT=${TRUNC_CKPT:-$HOME/w4a8_runs/rdna3_port/trunc/ckpt-L4}
RECIPE=tools/rdna/recipes/flashnext-w2-mtp2.env
E2E_DIR=${E2E_DIR:-$RUN}
E2E_MIN_FREE_GIB=130

# --- helpers -----------------------------------------------------------------
say() { printf '[%s] %s\n' "$(date +%T)" "$*"; }

junction_temp_c() {
    local h
    for h in /sys/bus/pci/devices/"$PCI_ADDR"/hwmon/hwmon*; do
        if [[ -r $h/temp2_input ]]; then echo $(($(cat "$h/temp2_input") / 1000)); return; fi
    done
    echo 0
}

touch_quiet() {
    mkdir -p "$LANES/quiet"
    touch "$QUIET"
    LAST_QUIET_TOUCH=$SECONDS
}

# KFD gpu_id of the W7800 (topology location_id is bus << 8), set in main.
KFD_GPU_ID=
kfd_gpu_id() {
    local node
    for node in /sys/class/kfd/kfd/topology/nodes/*; do
        if [[ $(awk '/^location_id/ {print $2}' "$node/properties" 2>/dev/null) == \
            $((0x$W7800_BUS << 8)) ]]; then
            cat "$node/gpu_id"
            return
        fi
    done
}

# eviction_jump: returns 0 when the evicted_ms of any process on the W7800 grew
# since the previous call (a first sighting counts when it is already non-zero).
declare -A EVICTED_PREV=()
eviction_jump() {
    local f v jump=1
    for f in /sys/class/kfd/kfd/proc/*/stats_"$KFD_GPU_ID"/evicted_ms; do
        [[ -r $f ]] || continue
        v=$(<"$f") || continue
        if [[ -n ${EVICTED_PREV[$f]:-} ]]; then
            if ((v > EVICTED_PREV[$f])); then jump=0; fi
        elif ((v > 0)); then
            jump=0
        fi
        EVICTED_PREV[$f]=$v
    done
    return "$jump"
}

# wait_window: a measured window needs the quiet file, a free boot.lock and no docker
# build; it is closed after 14 minutes (the rule is 15) and reopened on demand.
WINDOW_START=0
wait_window() {
    local waited=0
    if ((WINDOW_START && SECONDS - WINDOW_START > 840)); then
        rm -f "$QUIET"
        WINDOW_START=0
    fi
    touch_quiet
    until flock -n "$HOME/w4a8_runs/boot.lock" true &&
        ! pgrep -f '^(/usr/sbin/|/usr/bin/)?docker buildx build' >/dev/null; do
        sleep 10
        waited=$((waited + 10))
        touch_quiet
        if ((waited >= 1200)); then
            echo "boot.lock or a docker build was busy for 20 min" >&2
            return 1
        fi
    done
    if ((!WINDOW_START)); then WINDOW_START=$SECONDS; fi
}

# label_wanted LABEL: with --labels A,B only the listed commands run (a re-run of a few
# groups after a fix); without it everything selected by --from/--to/--only runs.
label_wanted() {
    local label=${1%-retry}
    [[ -z $LABELS || ,$LABELS, == *,"$label",* ]]
}

# run_cmd LABEL TIMEOUT_S QUIET(0|1) ARGV... : logs to $RUN/LABEL.log, samples the
# temperature every second, and in a measured window (QUIET=1) re-touches the quiet
# file and watches the KFD eviction counters. Returns the exit status of the command
# (124 timeout, 125 thermal abort, 123 clean exit but an eviction jump: the window
# is INVALID).
run_cmd() {
    local label=$1 limit=$2 quiet=$3
    shift 3
    if ! label_wanted "$label"; then return 0; fi
    if ((DRY)); then
        printf '%-26s ' "$label"
        printf '%q ' "$@"
        printf '\n'
        return 0
    fi
    local log=$RUN/$label.log t0=$SECONDS
    say "$label: start (timeout ${limit}s)"
    local watch=0
    if ((quiet)) && [[ -n $KFD_GPU_ID ]]; then
        watch=1
        EVICTED_PREV=()
        eviction_jump || true
    fi
    setsid timeout --kill-after=20 "$limit" "$@" >"$log" 2>&1 &
    local pid=$! rc=0 hot=0 evicted=0 maxt=0 t win=() sum i tick=0 flags=
    while kill -0 "$pid" 2>/dev/null; do
        t=$(junction_temp_c)
        if ((t > maxt)); then maxt=$t; fi
        win+=("$t")
        ((${#win[@]} > 5)) && win=("${win[@]:1}")
        sum=0
        for i in "${win[@]}"; do sum=$((sum + i)); done
        tick=$((tick + 1))
        ((tick % 5 == 0)) && echo "$(date +%T) $t" >>"$RUN/temp.log"
        if ((t >= 95 || (${#win[@]} == 5 && sum / 5 >= 90))); then
            hot=1
            say "$label: THERMAL ABORT at $t C"
            kill -TERM -- "-$pid" 2>/dev/null
            break
        fi
        if ((watch)) && eviction_jump; then evicted=1; fi
        if ((quiet && SECONDS - LAST_QUIET_TOUCH >= 240)); then touch_quiet; fi
        sleep 1
    done
    wait "$pid"
    rc=$?
    if ((hot)); then
        rc=125
    elif ((evicted && rc == 0)); then
        rc=123
    fi
    if ((evicted)); then flags=$'\tEVICTION-INVALID-WINDOW'; fi
    printf '%s\t%s\t%s\t%ss\tmaxT=%sC%s\n' "$label" "$rc" "$(date +%T)" \
        "$((SECONDS - t0))" "$maxt" "$flags" >>"$RUN/summary.tsv"
    say "$label: exit $rc in $((SECONDS - t0)) s, junction max $maxt C (log $log)"
    return "$rc"
}

# pytest group: fails when nothing passed (everything skipped = op or arch missing).
# Stops at the first failing case, unless --keep-going (every failing case is listed).
# VLLM_RDNA3_HIP_OPS=all: the tests of the gated wrappers (mrope_forward_rdna2, ...) were
# written for gfx10x, where the HIP path is always on; gfx1100 opts in per family.
pytest_group() {
    local label=$1 limit=$2 stop_flag=(-x)
    shift 2
    if ! label_wanted "$label"; then return 0; fi
    if ((KEEP_GOING)); then stop_flag=(); fi
    # "~" stands for a space inside one argument (a -k expression).
    local pytest_args=("${@//\~/ }")
    run_cmd "$label" "$limit" 0 "${GPU_ENV[@]}" VLLM_RDNA3_HIP_OPS=all \
        ${PYTEST_EXTRA_ENV[@]+"${PYTEST_EXTRA_ENV[@]}"} "$PY" -m pytest -q \
        "${stop_flag[@]}" -rfEs --tb=short -p no:cacheprovider "${pytest_args[@]}" || return $?
    if ((DRY)); then return 0; fi
    local line passed
    line=$(grep -E '[0-9]+ (passed|failed|skipped|error)' "$RUN/$label.log" | tail -1)
    passed=$(grep -oE '[0-9]+ passed' <<<"$line" | grep -oE '[0-9]+')
    say "$label: $line"
    if [[ -z $passed || $passed -eq 0 ]]; then
        say "$label: FAIL, no test passed (everything skipped?)"
        printf '%s\tNOPASS\n' "$label" >>"$RUN/summary.tsv"
        return 1
    fi
}

finish() {
    local rc=${1:-0}
    if ((!DRY)); then
        say "summary ($RUN/summary.tsv):"
        cat "$RUN/summary.tsv" 2>/dev/null
        rm -f "$LEASE" "$QUIET"
    fi
    exit "$rc"
}

# stop_or_go RC WHAT: records a failure and stops the run unless --keep-going.
stop_or_go() {
    if (($1 != 0)); then
        FAILED=1
        say "FAILED: $2 (exit $1)"
        if ((!KEEP_GOING)); then finish 1; fi
    fi
    return 0
}

acquire_lease() {
    if ((DRY)); then return 0; fi
    mkdir -p "$LANES/lease" "$RUN"
    local other
    for other in "$LANES"/lease/*.w7800cb; do
        if [[ -e $other && $other != "$LEASE" ]]; then
            echo "W7800 cb lease held by $other" >&2
            exit 3
        fi
    done
    printf '%s %s gpu_g0_g7.sh %s\n' "$$" "$(date -Is)" "${ONLY:-$FROM..$TO}" >"$LEASE"
    trap 'rm -f "$LEASE" "$QUIET"' EXIT
    say "lease $LEASE taken, logs in $RUN"
}

run_groups() {
    local step=$1 limit=$2 i=0 group
    shift 2
    for group in "$@"; do
        i=$((i + 1))
        # shellcheck disable=SC2086
        pytest_group "$step-$i" "$limit" $group
        stop_or_go $? "$step group $i: $group"
    done
}

# --- steps -------------------------------------------------------------------
step_G0() {
    say "G0: HIP $HIP_INDEX must be the W7800 on bus $W7800_BUS (sysfs first, then torch)"
    if ((!DRY)); then
        local dev=/sys/bus/pci/devices/$PCI_ADDR used total
        if [[ $(cat "$dev/vendor") != 0x1002 || $(cat "$dev/device") != 0x7449 ]]; then
            echo "PCI $PCI_ADDR is not a W7800 (0x1002:0x7449)" >&2
            return 1
        fi
        used=$(cat "$dev/mem_info_vram_used" 2>/dev/null || echo 0)
        total=$(cat "$dev/mem_info_vram_total" 2>/dev/null || echo 0)
        say "G0: vram used $((used >> 20)) of $((total >> 20)) MiB, junction $(junction_temp_c) C"
        if ((used > 2 * 1024 * 1024 * 1024)); then
            echo "W7800 cb is in use (vram used > 2 GiB)" >&2
            return 1
        fi
    fi
    run_cmd G0-preflight 300 0 "${GPU_ENV[@]}" "$PY" tools/rdna/rdna3/gpu_preflight.py \
        --bus "$W7800_BUS" --arch gfx1100 || return $?
    run_cmd G0-families 600 0 "${GPU_ENV[@]}" "$PY" tools/rdna/rdna3/gpu_preflight.py \
        --bus "$W7800_BUS" --arch gfx1100 --families
}

step_G1() {
    say "G1: build check (CPU only)"
    # shellcheck disable=SC2016
    run_cmd G1-sha 60 0 bash -c 'cat "$1/.sha"; cat "$1/overlay.txt" 2>/dev/null; true' \
        _ "$TREE" || return $?
    run_cmd G1-load 300 0 "${CPU_ENV[@]}" "$PY" tools/rdna/rdna3/rocm_c_load_check.py check \
        "$TREE/vllm/_rocm_C.abi3.so" --out "$RUN/ops.json" || return $?
    run_cmd G1-extensions 120 0 ls -la "$TREE"/vllm/_C.abi3.so "$TREE"/vllm/_rocm_C.abi3.so \
        "$TREE"/vllm/_C_stable_libtorch.abi3.so "$TREE"/vllm/_moe_C_stable_libtorch.abi3.so
}

step_G2() {
    say "G2: GDN prefill/decode and conv1d parity; delta-h against the fp64 oracle"
    run_groups G2 1800 "${G2_GROUPS[@]}"
    # HIP and Triton delta-h against the exact recurrence on the same inputs; the
    # dump of the first case is the input of a replay on another arch.
    run_cmd G2-oracle-failing 900 0 "${GPU_ENV[@]}" "$PY" \
        tools/rdna/rdna3/gdn_delta_h_oracle.py --cases failing \
        --dump "$RUN/delta_h_failing_case.pt"
    stop_or_go $? "G2 delta-h fp64 oracle, TP1 outlier case"
    run_cmd G2-oracle-sweep 1800 0 "${GPU_ENV[@]}" "$PY" \
        tools/rdna/rdna3/gdn_delta_h_oracle.py --cases sweep
    stop_or_go $? "G2 delta-h fp64 oracle, test grid"
}
step_G3() { say "G3: dense GEMV, QSA, HC, shared-expert parity"; run_groups G3 1800 "${G3_GROUPS[@]}"; }

step_G4() {
    say "G4: MoE (scalar kernel, Q32, unquantized HIP, WMMA when present)"
    run_groups G4 2400 "${G4_GROUPS[@]}"
    local stage i=0
    for stage in "${G4_WMMA[@]}"; do
        i=$((i + 1))
        if [[ -e $TREE/${stage%% *} ]]; then
            # shellcheck disable=SC2086
            pytest_group "G4-wmma-$i" 2400 $stage
            stop_or_go $? "G4 WMMA stage $i: $stage"
        else
            say "G4: ${stage%% *} is not in this tree, skipped"
        fi
    done
}

step_G5() { say "G5: upstream skinny GEMMs and top-k decode"; run_groups G5 1800 "${G5_GROUPS[@]}"; }
step_G6() { say "G6: widened V1 ops"; run_groups G6 1800 "${G6_GROUPS[@]}"; }

# measured_cell LABEL SCRIPT ARGS EXTRA_ENV: one measured cell, retried once when the
# window shows a KFD eviction jump.
measured_cell() {
    local label=$1 script=$2 args=$3 extra=$4 rc extra_arg=()
    if ! label_wanted "$label"; then return 0; fi
    read -ra extra_arg <<<"$extra"
    if ((!DRY)); then wait_window || return 1; fi
    # shellcheck disable=SC2086
    run_cmd "$label" 900 1 "${GPU_ENV[@]}" "${extra_arg[@]}" "$PY" "$script" $args
    rc=$?
    if ((rc == 123)); then
        say "$label: eviction jump in the window, one retry"
        wait_window || return 1
        # shellcheck disable=SC2086
        run_cmd "$label-retry" 900 1 "${GPU_ENV[@]}" "${extra_arg[@]}" "$PY" "$script" $args
        rc=$?
    fi
    return "$rc"
}

step_G7() {
    say "G7: per-op latency (measured windows); model first"
    run_cmd G7-model 120 0 "${CPU_ENV[@]}" "$PY" tools/rdna/rdna3/perop_expectations.py \
        --scenario central || return $?
    local cell label script args extra
    for cell in "${G7_CELLS[@]}"; do
        IFS='|' read -r label script args extra <<<"$cell"
        measured_cell "G7-$label" "$script" "$args" "$extra"
        stop_or_go $? "G7 cell $label"
    done
}

step_G8() {
    say "G8: top_k_per_row_decode, radix against insertion (measured window)"
    measured_cell G8-topk tools/rdna/rdna3/topk_decode_bench.py \
        "--top-k 2048 --batch 1,8,16 --next-n 3 --cols 1024,4096,8192,16384,32768,65536,131072" ""
    # The decode tests again on the insertion select: it is only worth a default (or a
    # recipe line) if they pass on it as well.
    PYTEST_EXTRA_ENV=(VLLM_TOPK_DECODE_ALGO=insertion)
    # shellcheck disable=SC2086
    pytest_group G8-insertion-tests 1800 $G5_TOPK
    stop_or_go $? "G8 decode top-k tests on the insertion select"
    PYTEST_EXTRA_ENV=()
}

# recipe_env FILE: the kernel-selecting variables of a recipe as NAME=value lines (plain
# values only; the cache root and the family list are set per run).
recipe_env() {
    local line re='^((VLLM|HSA|GPU|OMP|TOKENIZERS|PYTHONFAULTHANDLER)[A-Z0-9_]*)=([A-Za-z0-9_.,:/-]*)$'
    while IFS= read -r line; do
        [[ $line =~ $re ]] || continue
        case ${BASH_REMATCH[1]} in VLLM_CACHE_ROOT | VLLM_RDNA3_HIP_OPS) continue ;; esac
        printf '%s=%s\n' "${BASH_REMATCH[1]}" "${BASH_REMATCH[3]}"
    done <"$1"
}

# e2e_run LABEL DUMP [NAME=value...]: one engine start on the truncated checkpoint.
e2e_run() {
    local label=$1 dump=$2
    shift 2
    run_cmd "$label" 2700 0 "${GPU_ENV[@]}" ${E2E_ENV[@]+"${E2E_ENV[@]}"} "$@" "$PY" \
        tools/rdna/rdna3/trunc_e2e.py run --model "$TRUNC_CKPT" --out "$dump" --ppl \
        --max-model-len 8192
}

step_G13() {
    say "G13: truncated checkpoint end to end (base, every family, every family + WMMA)"
    local E2E_ENV=()
    if [[ -r $RECIPE ]]; then mapfile -t E2E_ENV < <(recipe_env "$RECIPE"); fi
    if ((!DRY)); then
        local free_gib
        free_gib=$(awk '/^MemAvailable:/ {print int($2 / 1048576)}' /proc/meminfo)
        if [[ ! -d $TRUNC_CKPT ]] || ((${#E2E_ENV[@]} == 0)); then
            echo "G13 needs the checkpoint $TRUNC_CKPT and the recipe $RECIPE" >&2
            return 1
        fi
        if ((free_gib < E2E_MIN_FREE_GIB)); then
            echo "G13 pins the 95 GiB PLE table: $free_gib GiB available, need $E2E_MIN_FREE_GIB" >&2
            return 1
        fi
    fi
    e2e_run G13-base "$E2E_DIR/e2e-base.pt"
    stop_or_go $? "G13 baseline run"
    e2e_run G13-all "$E2E_DIR/e2e-all.pt" VLLM_RDNA3_HIP_OPS=all
    stop_or_go $? "G13 every-family run"
    e2e_run G13-wmma "$E2E_DIR/e2e-wmma.pt" VLLM_RDNA3_HIP_OPS=all VLLM_RDNA3_MOE_WMMA=1
    stop_or_go $? "G13 WMMA run"
    # base against all: what the HIP families change; all against wmma: what the WMMA
    # kernel changes against the scalar one on the same inputs.
    run_cmd G13-compare-all 600 0 "${CPU_ENV[@]}" "$PY" tools/rdna/rdna3/trunc_e2e.py \
        compare "$E2E_DIR/e2e-base.pt" "$E2E_DIR/e2e-all.pt"
    stop_or_go $? "G13 comparison base / all"
    run_cmd G13-compare-wmma 600 0 "${CPU_ENV[@]}" "$PY" tools/rdna/rdna3/trunc_e2e.py \
        compare "$E2E_DIR/e2e-all.pt" "$E2E_DIR/e2e-wmma.pt"
    stop_or_go $? "G13 comparison all / wmma"
}

step_G16() {
    say "G16: gfx1100 rocBLAS / hipBLASLt Tensile libraries, fp16 and bf16 GEMMs"
    run_cmd G16-tensile 900 0 "${GPU_ENV[@]}" "$PY" tools/rdna/rdna3/tensile_probe.py \
        --arch gfx1100
}

# --- main --------------------------------------------------------------------
if ((LIST)); then
    for s in "${STEPS[@]}"; do if selected "$s"; then echo "$s"; fi; done
    exit 0
fi
if ! cd "$TREE" 2>/dev/null; then
    if ((!DRY)); then echo "no tree $TREE" >&2; exit 2; fi
fi
if ((!DRY)); then KFD_GPU_ID=$(kfd_gpu_id); fi
acquire_lease
for s in "${STEPS[@]}"; do
    selected "$s" || continue
    "step_$s"
    stop_or_go $? "$s"
done
finish "$FAILED"
