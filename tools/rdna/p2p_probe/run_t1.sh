#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# T1 driver (hetero-MoE design section 11, row T1): builds ipc_probe for
# gfx1030 + gfx1100, resolves HIP -> PCI bus, runs the single-process probes
# and the two-process hub/server probes in the requested visibility
# configurations, and prints a summary table. Runs on par1-cs25.
#
#   run_t1.sh [--hub-hip H] [--srv-hips A,B,C,D] [--tag T]
#             [--configs disjoint,shared,hub-own-first,srv-own-first]
#             [--subset v620] [--allow-slotb] [--tests LIST] [--quick]
#             [--timeout SECONDS] [--no-build] [--skip-single]
#             [--no-bus-check] [--dry-run] [-- EXTRA ipc_probe args]
#
# Defaults: hub HIP 0 (W7800), servers HIP 1,2,3,4 (V620, slot A), all four
# configurations. Visible sets per config (H = hub HIP, S = server HIP):
#   disjoint        hub {H}      server {S}
#   shared          hub {H,S}    server {H,S}   (server role device = ordinal 1)
#   hub-own-first   hub {H,S}    server {S}
#   srv-own-first   hub {H}      server {S,H}
# --subset v620: the V620-only run; the hub is the first V620 of --srv-hips
# (or --hub-hip if given) and the servers are the remaining V620s.
#
# Per config two sessions run: A (one server: ipc_open, store, lat, poll,
# alloc_kinds) and B (all servers: ipc_open, fanin). --tests restricts both.
#
# Allowed HIP indices: 0 and 1-4; 6-9 only with --allow-slotb; 5 never.
# Everything is built and logged under ~/w4a8_runs/<tag>/t1/.
#
# This script takes NO leases and NO boot.lock: the lead must hold the leases
# <Agent>.w7800cb and <Agent>.slotA (and respect quiet windows) before calling
# it. Every launch sets HIP_VISIBLE_DEVICES on its own command line and runs
# under timeout(1) plus a polling watchdog that kills only this script's
# children.
set -uo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROCM=${ROCM:-/opt/rocm/core-7.14}
HIPCC=${HIPCC:-$ROCM/bin/hipcc}
RUNS=${RUNS:-$HOME/w4a8_runs}
BUILD_LOCK=$RUNS/audit/build.lock

hub_hip=0
hub_set=0
srv_hips=1,2,3,4
tag=""
configs=disjoint,shared,hub-own-first,srv-own-first
subset=""
allow_slotb=0
tests_req=""
quick=0
timeout_s=1800
no_build=0
skip_single=0
no_bus_check=0
dry_run=0
extra=()

die() {
    echo "run_t1: $*" >&2
    exit 2
}

usage() {
    sed -n '4,/^set -uo/p' "${BASH_SOURCE[0]}" | grep '^#' | cut -c3-
    exit "${1:-0}"
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --hub-hip) hub_hip=${2:?}; hub_set=1; shift 2 ;;
        --srv-hips) srv_hips=${2:?}; shift 2 ;;
        --tag) tag=${2:?}; shift 2 ;;
        --configs) configs=${2:?}; shift 2 ;;
        --subset) subset=${2:?}; shift 2 ;;
        --allow-slotb) allow_slotb=1; shift ;;
        --tests) tests_req=${2:?}; shift 2 ;;
        --quick) quick=1; shift ;;
        --timeout) timeout_s=${2:?}; shift 2 ;;
        --no-build) no_build=1; shift ;;
        --skip-single) skip_single=1; shift ;;
        --no-bus-check) no_bus_check=1; shift ;;
        --dry-run) dry_run=1; shift ;;
        -h | --help) usage 0 ;;
        --) shift; extra=("$@"); break ;;
        *) die "unknown argument: $1 (see --help)" ;;
    esac
done

[[ -n $tag ]] || tag=t1-$(date +%Y%m%d-%H%M%S)
[[ $tag =~ ^[A-Za-z0-9._-]+$ ]] || die "bad --tag: $tag"
[[ $timeout_s =~ ^[0-9]+$ ]] || die "--timeout must be an integer"
[[ -z $subset || $subset == v620 ]] || die "--subset only knows v620"

# ---- device policy ----------------------------------------------------------
# Allowed: 0 (W7800 cb) and 1-4 (V620 slot A); 6-9 (V620 slot B) only with
# --allow-slotb; 5 (W7800 07) never.
check_hip() {
    local i=$1
    [[ $i =~ ^[0-9]+$ ]] || die "HIP index is not a number: '$i'"
    case $i in
        0 | 1 | 2 | 3 | 4) return 0 ;;
        5) die "REFUSED: HIP 5 (W7800 bus 07) must never be used" ;;
        6 | 7 | 8 | 9)
            [[ $allow_slotb -eq 1 ]] ||
                die "REFUSED: HIP $i is slot B; pass --allow-slotb to use it"
            ;;
        *) die "REFUSED: HIP $i is not in the allowed set (0, 1-4; 6-9 with --allow-slotb)" ;;
    esac
}

# Expected PCI bus and arch per HIP index (box rules, 2026-11-07).
expected_bus() {
    case $1 in
        0) echo cb ;; 1) echo ce ;; 2) echo d3 ;; 3) echo d8 ;; 4) echo db ;;
        6) echo 0a ;; 7) echo 0f ;; 8) echo 14 ;; 9) echo 17 ;;
    esac
}
expected_arch() {
    if [[ $1 -eq 0 ]]; then echo gfx1100; else echo gfx1030; fi
}

join_by_comma() {
    local IFS=,
    echo "$*"
}

IFS=, read -r -a SRV <<<"$srv_hips"
[[ ${#SRV[@]} -ge 1 ]] || die "--srv-hips is empty"
[[ ${#SRV[@]} -le 8 ]] || die "at most 8 servers"

if [[ $subset == v620 ]]; then
    if [[ $hub_set -eq 0 ]]; then
        hub_hip=${SRV[0]}
        SRV=("${SRV[@]:1}")
    else
        keep=()
        for s in "${SRV[@]}"; do [[ $s == "$hub_hip" ]] || keep+=("$s"); done
        SRV=("${keep[@]}")
    fi
    [[ ${#SRV[@]} -ge 1 ]] || die "--subset v620 needs at least two V620 indices"
fi

check_hip "$hub_hip"
for s in "${SRV[@]}"; do
    check_hip "$s"
    [[ $s != "$hub_hip" ]] || die "hub HIP $hub_hip is also in the server list"
done
if [[ $subset == v620 ]]; then
    [[ $hub_hip -ne 0 ]] || die "--subset v620 must not use the W7800 (HIP 0) as hub"
    for s in "${SRV[@]}"; do
        [[ $s -ne 0 ]] || die "--subset v620 must not use the W7800 (HIP 0) as server"
    done
fi
for ((a = 0; a < ${#SRV[@]}; a++)); do
    for ((b = a + 1; b < ${#SRV[@]}; b++)); do
        [[ ${SRV[a]} != "${SRV[b]}" ]] || die "duplicate server HIP ${SRV[a]}"
    done
done

IFS=, read -r -a CFGS <<<"$configs"
for c in "${CFGS[@]}"; do
    case $c in
        disjoint | shared | hub-own-first | srv-own-first) ;;
        *) die "unknown config: $c" ;;
    esac
done

SESS_A_DEFAULT=ipc_open,store,lat,poll,alloc_kinds
SESS_B_DEFAULT=ipc_open,fanin
filter_tests() { # $1 requested csv (may be empty), $2 default csv
    if [[ -z $1 ]]; then
        echo "$2"
        return
    fi
    local out=() t r req dflt
    IFS=, read -r -a req <<<"$1"
    IFS=, read -r -a dflt <<<"$2"
    for t in "${dflt[@]}"; do
        for r in "${req[@]}"; do
            [[ $t == "$r" ]] && out+=("$t")
        done
    done
    join_by_comma ${out[@]+"${out[@]}"}
}
if [[ -n $tests_req ]]; then
    IFS=, read -r -a REQ <<<"$tests_req"
    for r in "${REQ[@]}"; do
        case $r in
            ipc_open | store | lat | poll | fanin | alloc_kinds) ;;
            *) die "unknown test: $r" ;;
        esac
    done
fi
TESTS_A=$(filter_tests "$tests_req" "$SESS_A_DEFAULT")
TESTS_B=$(filter_tests "$tests_req" "$SESS_B_DEFAULT")

OUT=$RUNS/$tag/t1
BIN=$OUT/ipc_probe

ALL_HIPS=("$hub_hip" "${SRV[@]}")

echo "== T1 plan: tag=$tag out=$OUT"
echo "   hub HIP $hub_hip ($(expected_arch "$hub_hip")), servers HIP ${SRV[*]}"
echo "   configs: $configs   sessions: A[1 server]=${TESTS_A:-none} B[${#SRV[@]} servers]=${TESTS_B:-none}"
echo "   REMINDER: the leases <Agent>.w7800cb and <Agent>.slotA are required and are"
echo "   NOT taken by this script (the lead holds them, with boot.lock and quiet files)."

# ---- build ------------------------------------------------------------------
build() {
    mkdir -p "$OUT"
    local cmd=(flock "$BUILD_LOCK" nice -n 19 "$HIPCC" -x hip -O3 -std=c++17
        -Wall -Wextra --offload-arch=gfx1030 --offload-arch=gfx1100
        -o "$BIN" "$HERE/ipc_probe.cpp" -pthread)
    echo "+ ${cmd[*]}"
    if [[ $dry_run -eq 1 ]]; then return 0; fi
    if ! "${cmd[@]}" >"$OUT/build.log" 2>&1; then
        cat "$OUT/build.log" >&2
        die "build failed (see $OUT/build.log)"
    fi
    echo "   built $BIN (log $OUT/build.log, $(grep -c -i warning "$OUT/build.log") warning lines)"
}
if [[ $no_build -eq 1 ]]; then
    [[ $dry_run -eq 1 || -x $BIN ]] || die "--no-build but $BIN does not exist"
else
    build
fi

# ---- watchdog and process helpers ---------------------------------------------
CHILDREN=()
# Kill only this script's own children on any exit (including INT/TERM).
trap 'for p in ${CHILDREN[@]+"${CHILDREN[@]}"}; do kill "$p" 2>/dev/null; done' EXIT
trap 'exit 130' INT TERM

# probe CMD: run `ipc_probe ARGS...` with HIP_VISIBLE_DEVICES=$1 in the
# foreground under timeout(1); stdout+stderr go to file $2.
run_single() { # vis logfile args...
    local vis=$1 log=$2
    shift 2
    echo "+ HIP_VISIBLE_DEVICES=$vis $BIN $*  > $log"
    if [[ $dry_run -eq 1 ]]; then return 0; fi
    env -u ROCR_VISIBLE_DEVICES -u CUDA_VISIBLE_DEVICES HIP_VISIBLE_DEVICES="$vis" \
        LD_LIBRARY_PATH="$ROCM/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
        timeout --kill-after=15 "$timeout_s" "$BIN" "$@" >"$log" 2>&1
}

# ---- HIP -> PCI bus -----------------------------------------------------------
# (no state kept: the table below is the record)
resolve_buses() {
    local i line bus arch exp_bus exp_arch bad=0
    echo "== HIP -> PCI bus (resolved by ipc_probe buses, one visible device at a time)"
    for i in "${ALL_HIPS[@]}"; do
        exp_bus=$(expected_bus "$i")
        exp_arch=$(expected_arch "$i")
        if [[ $dry_run -eq 1 ]]; then
            echo "   HIP $i -> (dry-run, not resolved) expected bus $exp_bus arch $exp_arch"
            continue
        fi
        run_single "$i" "$OUT/buses.$i.log" buses
        line=$(grep '^T1 buses' "$OUT/buses.$i.log" | head -n1)
        bus=$(awk '{for (k = 1; k <= NF; k++) if ($k ~ /^bus=/) print substr($k, 5)}' <<<"$line" |
            awk -F: '{print tolower($2)}')
        arch=$(awk '{for (k = 1; k <= NF; k++) if ($k ~ /^arch=/) print substr($k, 6)}' <<<"$line")
        local verdict=OK
        if [[ $bus != "$exp_bus" || $arch != "$exp_arch"* ]]; then
            verdict="MISMATCH (expected bus $exp_bus arch $exp_arch)"
            bad=1
        fi
        echo "   HIP $i -> bus ${bus:-?} arch ${arch:-?}  $verdict"
    done
    if [[ $bad -eq 1 && $no_bus_check -eq 0 ]]; then
        die "REFUSED: resolved HIP -> bus/arch differs from the box rules (use --no-bus-check only after the lead confirms the new order)"
    fi
}

# ---- sessions -----------------------------------------------------------------
LOGS=()
RCS=()
probe_args() { # common ipc_probe args
    local a=()
    [[ $quick -eq 1 ]] && a+=(--quick)
    printf '%s\n' ${a[@]+"${a[@]}"} ${extra[@]+"${extra[@]}"}
}

run_session() { # cfg sess nsrv tests-csv
    local cfg=$1 sess=$2 n=$3 tl=$4
    [[ -n $tl ]] || return 0
    local base=$OUT/$cfg.$sess rdvdir=$OUT/$cfg.$sess.rdv
    local srv=("${SRV[@]:0:n}")
    local hub_vis hub_dev=0 srv_dev i
    local srv_vis=()
    case $cfg in
        disjoint)
            hub_vis=$hub_hip
            srv_dev=0
            for s in "${srv[@]}"; do srv_vis+=("$s"); done
            ;;
        shared)
            hub_vis=$(join_by_comma "$hub_hip" "${srv[@]}")
            srv_dev=1
            for s in "${srv[@]}"; do srv_vis+=("$hub_hip,$s"); done
            ;;
        hub-own-first)
            hub_vis=$(join_by_comma "$hub_hip" "${srv[@]}")
            srv_dev=0
            for s in "${srv[@]}"; do srv_vis+=("$s"); done
            ;;
        srv-own-first)
            hub_vis=$hub_hip
            srv_dev=0
            for s in "${srv[@]}"; do srv_vis+=("$s,$hub_hip"); done
            ;;
    esac
    local hub_arch srv_arch
    hub_arch=$(expected_arch "$hub_hip")
    srv_arch=$(expected_arch "${srv[0]}")
    local common=()
    while IFS= read -r line; do [[ -n $line ]] && common+=("$line"); done < <(probe_args)
    echo "== session $cfg.$sess: hub vis {$hub_vis} dev $hub_dev; servers ${srv[*]} (vis ${srv_vis[*]}) dev $srv_dev; tests $tl"
    if [[ $dry_run -eq 0 ]]; then
        rm -rf "$rdvdir"
        mkdir -p "$rdvdir"
    fi
    local pids=() names=() logs=()
    for ((i = 0; i < n; i++)); do
        local log=$base.srv$i.log
        echo "+ HIP_VISIBLE_DEVICES=${srv_vis[i]} $BIN srv --dir $rdvdir --n-srv $n --id $i --dev $srv_dev --tests $tl --expect-arch $srv_arch --label $cfg.$sess ${common[*]-}  > $log"
        logs+=("$log")
        if [[ $dry_run -eq 0 ]]; then
            env -u ROCR_VISIBLE_DEVICES -u CUDA_VISIBLE_DEVICES HIP_VISIBLE_DEVICES="${srv_vis[i]}" \
                LD_LIBRARY_PATH="$ROCM/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
                timeout --kill-after=15 "$timeout_s" "$BIN" srv --dir "$rdvdir" \
                --n-srv "$n" --id "$i" --dev "$srv_dev" --tests "$tl" \
                --expect-arch "$srv_arch" --label "$cfg.$sess" \
                ${common[@]+"${common[@]}"} >"$log" 2>&1 &
            pids+=($!)
            CHILDREN+=($!)
            names+=("srv$i")
        fi
    done
    local hlog=$base.hub.log
    echo "+ HIP_VISIBLE_DEVICES=$hub_vis $BIN hub --dir $rdvdir --n-srv $n --dev $hub_dev --tests $tl --expect-arch $hub_arch --label $cfg.$sess ${common[*]-}  > $hlog"
    logs+=("$hlog")
    if [[ $dry_run -eq 0 ]]; then
        env -u ROCR_VISIBLE_DEVICES -u CUDA_VISIBLE_DEVICES HIP_VISIBLE_DEVICES="$hub_vis" \
            LD_LIBRARY_PATH="$ROCM/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
            timeout --kill-after=15 "$timeout_s" "$BIN" hub --dir "$rdvdir" \
            --n-srv "$n" --dev "$hub_dev" --tests "$tl" \
            --expect-arch "$hub_arch" --label "$cfg.$sess" \
            ${common[@]+"${common[@]}"} >"$hlog" 2>&1 &
        pids+=($!)
        CHILDREN+=($!)
        names+=(hub)
    else
        return 0
    fi
    # Watchdog: timeout(1) kills the probes at $timeout_s; this loop gives up
    # on children that survive SIGKILL (stuck in the driver) after +60 s.
    local deadline=$((SECONDS + timeout_s + 60)) running
    while true; do
        running=$(jobs -pr)
        [[ -n $running ]] || break
        if ((SECONDS > deadline)); then
            echo "WATCHDOG: children still running after $((timeout_s + 60)) s, killing: $(echo "$running" | tr '\n' ' ')" >&2
            for p in $running; do kill -KILL "$p" 2>/dev/null; done
            sleep 2
            break
        fi
        sleep 1
    done
    running=$(jobs -pr)
    for i in "${!pids[@]}"; do
        local rc
        if grep -qx "${pids[i]}" <<<"$running"; then
            rc=STUCK
        else
            wait "${pids[i]}"
            rc=$?
        fi
        RCS+=("$cfg.$sess ${names[i]} rc=$rc")
        echo "   $cfg.$sess ${names[i]} exit $rc"
    done
    LOGS+=("${logs[@]}")
    CHILDREN=()
}

# ---- run ------------------------------------------------------------------------
if [[ $dry_run -eq 0 ]]; then mkdir -p "$OUT"; fi
resolve_buses

if [[ $skip_single -eq 0 ]]; then
    sargs=()
    [[ $quick -eq 1 ]] && sargs+=(--tick-ms 200 --tick-reps 3)
    for i in "${ALL_HIPS[@]}"; do
        for cmd in info tick alloc; do
            run_single "$i" "$OUT/single.$cmd.$i.log" "$cmd" --label "single.hip$i" \
                ${sargs[@]+"${sargs[@]}"}
            rc=$?
            [[ $dry_run -eq 1 ]] || LOGS+=("$OUT/single.$cmd.$i.log")
            [[ $dry_run -eq 1 ]] || RCS+=("single.$cmd.$i - rc=$rc")
        done
    done
fi

for c in "${CFGS[@]}"; do
    run_session "$c" A 1 "$TESTS_A"
    run_session "$c" B "${#SRV[@]}" "$TESTS_B"
done

if [[ $dry_run -eq 1 ]]; then
    echo "== dry run: nothing was built, resolved or launched"
    exit 0
fi

# ---- summary --------------------------------------------------------------------
echo
echo "== SUMMARY (tag $tag; full lines in $OUT/*.log)"
printf '%-26s %-8s %-12s %-5s %s\n' label role test verdict metrics
if [[ ${#LOGS[@]} -gt 0 ]]; then
    grep -h '^T1 ' "${LOGS[@]}" | awk '
BEGIN {
    K["buses"] = "dev bus arch"
    K["info"] = "dev arch wall_clock_khz uncached finegrained"
    K["info_p2p"] = "pair access perf_rank native_atomics"
    K["tick"] = "dev arch attr_ticks_per_ms measured_ticks_per_ms err_pct use"
    K["alloc"] = "dev arch kind alloc memset ipc_export"
    K["hello"] = "dev arch bus measured_ticks_per_ms buffer_kind"
    K["ipc_open"] = "peer open ptr_device ptr_device_arch ptr_maps_peer_bus remote_write_bad_words remote_read_bad_words stage"
    K["store"] = "size iters_per_s mismatches first_bad timeouts srv_timeouts"
    K["lat"] = "size variant p50_us p99_us max_us timeouts"
    K["poll"] = "dir variant rtt_p50_us rtt_p99_us rtt_max_us slow_gt100x_median out_of_order timeouts"
    K["fanin"] = "what n p50_us p99_us agg_gbps_mean verify_bad_words timeouts"
    K["alloc_kinds"] = "kind coherent store_mismatches slow_total ooo_total timeouts_total"
}
{
    split("", kv)
    test = $2
    verdict = $NF
    for (i = 3; i < NF; i++) {
        eq = index($i, "=")
        if (eq > 0) kv[substr($i, 1, eq - 1)] = substr($i, eq + 1)
    }
    n = split(K[test], ks, " ")
    out = ""
    for (j = 1; j <= n; j++)
        if (ks[j] in kv) out = out ks[j] "=" kv[ks[j]] " "
    if (out == "") out = $0
    printf "%-26s %-8s %-12s %-5s %s\n", kv["label"], kv["role"], test, verdict, out
}'
fi
echo
echo "== process exits"
fail=0
for ((k = 0; k < ${#RCS[@]}; k++)); do
    echo "   ${RCS[k]}"
    [[ ${RCS[k]} == *"rc=0" ]] || fail=1
done
nfail=0
if [[ ${#LOGS[@]} -gt 0 ]]; then
    nfail=$(grep -h '^T1 ' "${LOGS[@]}" | grep -c ' FAIL$')
fi
echo "== FAIL lines: $nfail   non-zero exits: $fail"
if [[ $nfail -gt 0 || $fail -ne 0 ]]; then
    echo "== T1 RESULT: FAIL (see the table; logs in $OUT)"
    exit 1
fi
echo "== T1 RESULT: no FAIL lines (INFO rows still need reading; logs in $OUT)"
exit 0
