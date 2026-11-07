#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Driver of the hx loopback test (tools/rdna/hetero_moe/hx_loopback.cpp): builds
# the binary for gfx1030 + gfx1100, resolves HIP -> PCI bus, starts one hub and
# one server process per server HIP on their own GPUs, collects exit codes and
# the HXLB / HXBENCH lines, and prints a PASS / FAIL summary. Runs on par1-cs25.
#
#   run_loopback.sh [--hub-hip H] [--srv-hips A,B,C,D] [--tag T]
#                   [--visible disjoint|shared] [--timeout SECONDS]
#                   [--kill-test server|hub|hang] [--fault-after N]
#                   [--allow-slotb] [--build-only] [--no-build] [--rebuild]
#                   [--bin PATH] [--build-wait SECONDS] [--name N]
#                   [--rdv-root DIR] [--no-bus-check] [--dry-run]
#                   [-- EXTRA hx_loopback options for every process]
#
# Defaults: hub HIP 0 (W7800, gfx1100), servers HIP 1,2,3,4 (V620, slot A),
# --visible disjoint (every process sees only its own GPU, as the launcher of
# the runtime does), 1e6 exchanges. EXTRA options (--iters, --T-list, --graph,
# --eager, --bench, --nslots, --srv-work-us, --wait-ms, --no-watcher, ...) are
# appended last and win over the presets of the driver.
#
# --timeout bounds every process (timeout(1), default 3600 s); 1e6 exchanges may
# need more. The binary of a tag is reused while it is newer than its sources
# (--rebuild forces a build; --no-build and --bin PATH never build).
#
# --visible shared: every process sees all participating GPUs (hub first, then
# the servers) and selects its own with --device.
#
# --kill-test: a fault injection run (the survivors must end with exit 3 and the
# victim with 9; "out" is within 5 s of the fault with the watcher, else within
# the larger of --wait-ms and --begin-wait-ms plus 3 s):
#   server  server 0 _exit(9)s after --fault-after exchanges
#   hub     the hub _exit(9)s after --fault-after exchanges
#   hang    server 0 stops pushing replies after --fault-after exchanges: the hub
#           aborts in COMBINE after --wait-ms (3000) and every process exits 3
# With the watcher on, server and hub use --wait-ms 600000 --begin-wait-ms 600000
# so only the pidfd watcher can end the run early; with --no-watcher in EXTRA
# they use --wait-ms 3000 --begin-wait-ms 6000 and the device timeouts decide.
#
# Allowed HIP indices: 0 and 1-4; 6-9 only with --allow-slotb; 5 never.
# The binary is built under ~/w4a8_runs/<tag>/hxlb/ (or taken from --bin), logs
# of a run go to ~/w4a8_runs/<tag>/hxlb/<name>/, the rendezvous directory is
# <rdv-root>/hxlb_<tag>_<pid> (default /dev/shm, never /tmp).
#
# Exit: 0 PASS, 1 FAIL, 2 usage or policy error, 3 build or self-test failed,
# 4 a HIP index is not the expected GPU.
#
# This script takes NO leases and NO boot.lock: the lead must hold the leases
# <Agent>.w7800cb (HIP 0) and <Agent>.slotA (HIP 1-4) or <Agent>.slotB (HIP 6-9)
# before calling it, and a --bench run is a measured window (quiet file). Every
# launch sets HIP_VISIBLE_DEVICES on its own command line and runs under
# timeout(1) plus a polling watchdog that kills only this script's children.
set -uo pipefail
export LC_ALL=C

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$HERE/../../.." && pwd)
SRC=$HERE/hx_loopback.cpp
HX_INC=$REPO/csrc/rocm/hetero_moe
ROCM=${ROCM:-/opt/rocm/core-7.14}
HIPCC=${HIPCC:-$ROCM/bin/hipcc}
RUNS=${RUNS:-$HOME/w4a8_runs}
BUILD_LOCK=$RUNS/audit/build.lock

hub_hip=0
srv_hips=1,2,3,4
tag=""
visible=disjoint
timeout_s=3600
kill_test=""
fault_after=3000
allow_slotb=0
build_only=0
no_build=0
rebuild=0
bin_override=""
build_wait=3000
name=""
rdv_root=/dev/shm
no_bus_check=0
dry_run=0
extra=()

die() {
    echo "run_loopback: $*" >&2
    exit 2
}

die4() {
    echo "run_loopback: $*" >&2
    exit 4
}

die3() {
    echo "run_loopback: $*" >&2
    exit 3
}

note() { echo "[hxlb $(date +%T)] $*"; }

usage() {
    sed -n '4,/^set -uo/p' "${BASH_SOURCE[0]}" | grep '^#' | cut -c3-
    exit "${1:-0}"
}

# need OPTION ARGC: the option takes a value
need() { [[ $2 -ge 2 ]] || die "$1 needs a value"; }

while [[ $# -gt 0 ]]; do
    case $1 in
        --hub-hip) need "$1" $#; hub_hip=$2; shift 2 ;;
        --srv-hips) need "$1" $#; srv_hips=$2; shift 2 ;;
        --tag) need "$1" $#; tag=$2; shift 2 ;;
        --visible) need "$1" $#; visible=$2; shift 2 ;;
        --timeout) need "$1" $#; timeout_s=$2; shift 2 ;;
        --kill-test) need "$1" $#; kill_test=$2; shift 2 ;;
        --fault-after) need "$1" $#; fault_after=$2; shift 2 ;;
        --allow-slotb) allow_slotb=1; shift ;;
        --build-only) build_only=1; shift ;;
        --no-build) no_build=1; shift ;;
        --rebuild) rebuild=1; shift ;;
        --bin) need "$1" $#; bin_override=$2; shift 2 ;;
        --build-wait) need "$1" $#; build_wait=$2; shift 2 ;;
        --name) need "$1" $#; name=$2; shift 2 ;;
        --rdv-root) need "$1" $#; rdv_root=$2; shift 2 ;;
        --no-bus-check) no_bus_check=1; shift ;;
        --dry-run) dry_run=1; shift ;;
        -h | --help) usage 0 ;;
        --) shift; extra=("$@"); break ;;
        *) die "unknown argument: $1 (see --help)" ;;
    esac
done

[[ -n ${EPOCHREALTIME:-} ]] || die "needs bash >= 5 (EPOCHREALTIME)"
[[ -n $tag ]] || tag=hxlb-$(date +%Y%m%d-%H%M%S)
[[ $tag =~ ^[A-Za-z0-9._-]+$ ]] || die "bad --tag: $tag"
[[ $visible == disjoint || $visible == shared ]] || die "--visible must be disjoint or shared"
[[ -z $kill_test || $kill_test == server || $kill_test == hub || $kill_test == hang ]] ||
    die "--kill-test must be server, hub or hang"
[[ $timeout_s =~ ^[0-9]+$ && $timeout_s -ge 1 ]] || die "--timeout must be a positive integer"
[[ $fault_after =~ ^[0-9]+$ ]] || die "--fault-after must be an integer"
[[ $build_wait =~ ^[0-9]+$ ]] || die "--build-wait must be an integer"
[[ -z $name || $name =~ ^[A-Za-z0-9._-]+$ ]] || die "bad --name: $name"
[[ $rdv_root == /* && $rdv_root != *[[:space:]]* ]] ||
    die "--rdv-root must be an absolute path without blanks"
case $rdv_root/ in /tmp/* | /var/tmp/* | /private/tmp/*) die "--rdv-root must not be under /tmp" ;; esac
case $RUNS/ in /tmp/* | /var/tmp/* | /private/tmp/*) die "RUNS must not be under /tmp" ;; esac

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

IFS=, read -r -a srv_list <<<"$srv_hips"
nsrv=${#srv_list[@]}
[[ $nsrv -ge 1 && $nsrv -le 8 ]] || die "--srv-hips needs 1 to 8 entries"
check_hip "$hub_hip"
for ((i = 0; i < nsrv; i++)); do
    check_hip "${srv_list[i]}"
    for ((j = 0; j < i; j++)); do
        [[ ${srv_list[i]} != "${srv_list[j]}" ]] || die "server HIP ${srv_list[i]} listed twice"
    done
done

# ---- roles ------------------------------------------------------------------
# role 0 is the hub, role 1 + s is server s.
rname=(hub)
rhip=("$hub_hip")
for ((s = 0; s < nsrv; s++)); do
    rname+=("srv$s")
    rhip+=("${srv_list[s]}")
done
nroles=${#rname[@]}

# every participating HIP once, hub first (the visible list of --visible shared)
vis_list=()
for h in "${rhip[@]}"; do
    dup=0
    for v in "${vis_list[@]+"${vis_list[@]}"}"; do
        [[ $v == "$h" ]] && dup=1
    done
    [[ $dup -eq 1 ]] || vis_list+=("$h")
done
vis_all=$(
    IFS=,
    echo "${vis_list[*]}"
)

rvis=()
rdev=()
for ((r = 0; r < nroles; r++)); do
    if [[ $visible == shared ]]; then
        rvis+=("$vis_all")
        for ((k = 0; k < ${#vis_list[@]}; k++)); do
            [[ ${vis_list[k]} == "${rhip[r]}" ]] && rdev+=("$k")
        done
    else
        rvis+=("${rhip[r]}")
        rdev+=(0)
    fi
done

has_extra() {
    local a
    for a in "${extra[@]+"${extra[@]}"}"; do
        [[ $a == "$1" ]] && return 0
    done
    return 1
}

watcher=1
has_extra --no-watcher && watcher=0
bench=0
has_extra --bench && bench=1

# effective value of an EXTRA option (the last occurrence wins)
extra_opt() {
    local v=$2 i
    for ((i = 0; i + 1 < ${#extra[@]}; i++)); do
        [[ ${extra[i]} != "$1" ]] || v=${extra[i + 1]}
    done
    echo "$v"
}
wait_ms=$(extra_opt --wait-ms 3000)
begin_ms=$(extra_opt --begin-wait-ms 6000)
[[ $wait_ms =~ ^[0-9]+$ && $begin_ms =~ ^[0-9]+$ ]] ||
    die "--wait-ms and --begin-wait-ms need integer values"
# how long after the fault the survivors may take to exit
if [[ $kill_test == hang || $watcher -eq 0 ]]; then
    limit_ms=$(((wait_ms > begin_ms ? wait_ms : begin_ms) + 3000))
else
    limit_ms=5000
fi

if [[ -z $name ]]; then
    if [[ -n $kill_test ]]; then
        name=kill-$kill_test
    elif [[ $bench -eq 1 ]]; then
        name=bench
    else
        name=soak
    fi
fi

OUT=$RUNS/$tag/hxlb
BIN=${bin_override:-$OUT/hx_loopback}
LOGDIR=$OUT/$name
RDV=$rdv_root/hxlb_${tag}_$$

note "tag $tag; hub HIP $hub_hip, servers HIP $srv_hips; visibility $visible; kill-test ${kill_test:-none}"
for ((s = 0; s < nsrv; s++)); do
    [[ ${srv_list[s]} != "$hub_hip" ]] || note "NOTE: server $s shares HIP $hub_hip with the hub"
done
if [[ $build_only -eq 0 ]]; then
    note "this script takes NO leases and NO boot.lock; before it runs the lead must hold:"
    need_w7800=0
    need_a=0
    need_b=0
    for h in "${vis_list[@]}"; do
        if [[ $h -eq 0 ]]; then
            need_w7800=1
        elif [[ $h -le 4 ]]; then
            need_a=1
        else
            need_b=1
        fi
    done
    [[ $need_w7800 -eq 0 ]] || note "  <Agent>.w7800cb   (HIP 0, W7800 bus cb)"
    [[ $need_a -eq 0 ]] || note "  <Agent>.slotA     (HIP 1-4, V620 ce d3 d8 db)"
    [[ $need_b -eq 0 ]] || note "  <Agent>.slotB     (HIP 6-9, V620 0a 0f 14 17)"
    [[ $bench -eq 0 ]] || note "  a --bench run is a measured window: touch lanes/quiet/<Agent> first"
fi

# ---- build ------------------------------------------------------------------
build() {
    local cmd=(flock -w "$build_wait" "$BUILD_LOCK" nice -n 19 "$HIPCC"
        --offload-arch=gfx1030 --offload-arch=gfx1100 -O3 -std=c++17
        -Wall -Wextra -x hip -I "$HX_INC" -o "$BIN" "$SRC" -pthread)
    echo "+ ${cmd[*]}  > $OUT/build.log"
    [[ $dry_run -eq 0 ]] || return 0
    command -v flock >/dev/null || die3 "flock not found"
    [[ -x $HIPCC ]] || die3 "hipcc not found: $HIPCC"
    mkdir -p "$OUT" "$RUNS/audit" || die3 "cannot create $OUT"
    note "waiting for the build lock $BUILD_LOCK (up to $build_wait s), then compiling"
    if ! "${cmd[@]}" >"$OUT/build.log" 2>&1; then
        tail -n 40 "$OUT/build.log"
        echo "run_loopback: build failed (an empty log means no build lock within $build_wait s): $OUT/build.log" >&2
        exit 3
    fi
    note "built $BIN; $(grep -c 'warning:' "$OUT/build.log") warning line(s) in $OUT/build.log"
}

if [[ -n $bin_override ]]; then
    [[ $dry_run -eq 1 || -x $BIN ]] || die "--bin $BIN is not an executable file"
elif [[ $no_build -eq 1 ]]; then
    [[ $dry_run -eq 1 || -x $BIN ]] || die "--no-build: $BIN does not exist"
elif [[ $rebuild -eq 0 && -x $BIN && $BIN -nt $SRC && $BIN -nt $HX_INC/hx_core.cuh ]]; then
    note "reusing $BIN (newer than its sources; --rebuild forces a build)"
else
    build
fi
[[ $build_only -eq 0 ]] || exit 0

# The CPU-only part of the binary (reference model self-test): no GPU touched.
echo "+ HIP_VISIBLE_DEVICES= $BIN selftest"
if [[ $dry_run -eq 0 ]]; then
    if ! env -u ROCR_VISIBLE_DEVICES -u CUDA_VISIBLE_DEVICES HIP_VISIBLE_DEVICES= \
        LD_LIBRARY_PATH="$ROCM/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
        "$BIN" selftest; then
        echo "run_loopback: the CPU reference self-test failed" >&2
        exit 3
    fi
fi

# ---- HIP -> bus ---------------------------------------------------------------
# bus_line OUTPUT IDX -> the HIPBUS line of in-process ordinal IDX
bus_line() {
    local l
    while IFS= read -r l; do
        if [[ $l == "HIPBUS idx=$2 "* ]]; then
            echo "$l"
            return 0
        fi
    done <<<"$1"
    return 1
}

# buses VISIBLE -> the HIPBUS lines of a process that sees VISIBLE
buses() {
    env -u ROCR_VISIBLE_DEVICES -u CUDA_VISIBLE_DEVICES HIP_VISIBLE_DEVICES="$1" \
        LD_LIBRARY_PATH="$ROCM/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
        "$BIN" buses 2>&1
}

bus_check() {
    local r out shared_out="" line bus arch ok=1 want_bus want_arch
    printf '%-6s %-5s %-5s %-9s %s\n' role HIP bus arch device
    for ((r = 0; r < nroles; r++)); do
        if [[ $visible == shared ]]; then
            [[ -n $shared_out ]] || shared_out=$(buses "$vis_all") || {
                echo "$shared_out" >&2
                die4 "hx_loopback buses failed"
            }
            out=$shared_out
        else
            out=$(buses "${rhip[r]}") || {
                echo "$out" >&2
                die4 "hx_loopback buses failed for HIP ${rhip[r]}"
            }
        fi
        line=$(bus_line "$out" "${rdev[r]}") || {
            echo "$out" >&2
            die4 "no HIPBUS line for ordinal ${rdev[r]} (HIP ${rhip[r]})"
        }
        bus='?'
        arch='?'
        [[ $line =~ bus=[0-9a-fA-F]+:([0-9a-fA-F]+): ]] && bus=${BASH_REMATCH[1],,}
        [[ $line =~ arch=([^[:space:]:]+) ]] && arch=${BASH_REMATCH[1]}
        want_bus=$(expected_bus "${rhip[r]}")
        want_arch=$(expected_arch "${rhip[r]}")
        printf '%-6s %-5s %-5s %-9s %s\n' "${rname[r]}" "${rhip[r]}" "$bus" "$arch" "${rdev[r]}"
        if [[ $bus != "$want_bus" || $arch != "$want_arch" ]]; then
            echo "  ^ expected bus $want_bus arch $want_arch" >&2
            ok=0
        fi
    done
    [[ $ok -eq 1 ]]
}

if [[ $dry_run -eq 1 ]]; then
    note "dry run: the bus check is skipped"
elif [[ $no_bus_check -eq 1 ]]; then
    note "bus check skipped (--no-bus-check)"
else
    bus_check || {
        echo "run_loopback: a HIP index is not the expected GPU (box rules: HIP -> bus before every launch)" >&2
        exit 4
    }
fi

# ---- the run ------------------------------------------------------------------
# Presets first, EXTRA last (the last occurrence of an option wins).
common="--nsrv $nsrv"
srv0_preset=""
hub_preset=""
if [[ -n $kill_test ]]; then
    if [[ $kill_test == hang || $watcher -eq 0 ]]; then
        common+=" --wait-ms 3000 --begin-wait-ms 6000"
    else
        common+=" --wait-ms 600000 --begin-wait-ms 600000"
    fi
    common+=" --iters $((fault_after * 20 + 1000))"
    case $kill_test in
        server) srv0_preset="--die-after $fault_after" ;;
        hang) srv0_preset="--hang-after $fault_after" ;;
        hub) hub_preset="--hub-die-after $fault_after" ;;
    esac
fi

rcmd=("hub --dir $RDV --device ${rdev[0]} $common $hub_preset")
for ((s = 0; s < nsrv; s++)); do
    pre=""
    [[ $s -ne 0 ]] || pre=$srv0_preset
    rcmd+=("srv --dir $RDV --index $s --device ${rdev[s + 1]} $common $pre")
done

rpid=()
rrc=()
rend=()
cleanup_done=0
cleanup() {
    [[ $cleanup_done -eq 0 ]] || return 0
    cleanup_done=1
    local p any=0
    for p in "${rpid[@]+"${rpid[@]}"}"; do
        if [[ -n $p ]] && kill -0 "$p" 2>/dev/null; then
            kill -TERM "$p" 2>/dev/null
            any=1
        fi
    done
    if [[ $any -eq 1 ]]; then
        sleep 2
        for p in "${rpid[@]+"${rpid[@]}"}"; do
            if [[ -n $p ]] && kill -0 "$p" 2>/dev/null; then kill -KILL "$p" 2>/dev/null; fi
        done
    fi
    [[ -d $RDV ]] || return 0
    if mkdir -p "$LOGDIR/rdv"; then cp -a "$RDV"/. "$LOGDIR/rdv"/ 2>/dev/null; fi
    rm -rf "$RDV"
}
on_signal() {
    echo "run_loopback: interrupted, stopping the children" >&2
    cleanup
    exit 130
}
trap cleanup EXIT
trap on_signal INT TERM HUP

launch() {
    local r=$1 log=$LOGDIR/${rname[$1]}.log
    local -a args
    read -r -a args <<<"${rcmd[r]}"
    echo "+ HIP_VISIBLE_DEVICES=${rvis[r]} $BIN ${args[*]} ${extra[*]-}  > $log"
    [[ $dry_run -eq 0 ]] || return 0
    echo "+ HIP_VISIBLE_DEVICES=${rvis[r]} $BIN ${args[*]} ${extra[*]-}" >"$log"
    env -u ROCR_VISIBLE_DEVICES -u CUDA_VISIBLE_DEVICES \
        HIP_VISIBLE_DEVICES="${rvis[r]}" \
        LD_LIBRARY_PATH="$ROCM/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
        timeout --kill-after=15 "$timeout_s" "$BIN" "${args[@]}" "${extra[@]+"${extra[@]}"}" \
        >>"$log" 2>&1 &
    rpid[r]=$!
    rrc[r]=""
    rend[r]=""
}

if [[ $dry_run -eq 0 ]]; then
    mkdir -p "$LOGDIR" "$RDV" || die "cannot create $LOGDIR or $RDV"
else
    echo "(dry run: nothing is started)"
fi
t_start=$EPOCHREALTIME
# servers first (they wait for the hub's file), then the hub
for ((r = 1; r < nroles; r++)); do launch "$r"; done
launch 0
[[ $dry_run -eq 0 ]] || exit 0

note "started $nroles processes; logs in $LOGDIR"
deadline=$((SECONDS + timeout_s + 60))
while :; do
    alive=0
    for ((r = 0; r < nroles; r++)); do
        [[ -z ${rend[r]} ]] || continue
        if kill -0 "${rpid[r]}" 2>/dev/null; then
            alive=1
        else
            wait "${rpid[r]}"
            rrc[r]=$?
            rend[r]=$EPOCHREALTIME
        fi
    done
    [[ $alive -eq 1 ]] || break
    if ((SECONDS > deadline)); then
        echo "run_loopback: watchdog: a process outlived the timeout, stopping the children" >&2
        cleanup
        for ((r = 0; r < nroles; r++)); do
            [[ -z ${rend[r]} ]] || continue
            wait "${rpid[r]}" 2>/dev/null
            rrc[r]=$?
            rend[r]=$EPOCHREALTIME
        done
        break
    fi
    sleep 0.2
done

# ---- summary ------------------------------------------------------------------
# field LINE KEY -> value of KEY=value in an HXLB line, or -
field() {
    if [[ $1 =~ (^|[[:space:]])$2=([^[:space:]]*) ]]; then
        echo "${BASH_REMATCH[2]}"
    else
        echo -
    fi
}

verdict=PASS
fails=()
fail() {
    verdict=FAIL
    fails+=("$*")
}

# the injected victim leaves through _exit(9) without an HXLB line
victim=-1
case $kill_test in
    server) victim=1 ;;
    hub) victim=0 ;;
esac

hxlb=()
for ((r = 0; r < nroles; r++)); do
    hxlb+=("$(grep -h '^HXLB ' "$LOGDIR/${rname[r]}.log" 2>/dev/null | tail -n 1)")
done

echo
printf '%-5s %-4s %-5s %-7s %-9s %-11s %-12s %-14s %-5s %s\n' \
    role HIP rc wall_s iters mismatches reason phase peer k2exit_ms
for ((r = 0; r < nroles; r++)); do
    l=${hxlb[r]}
    wall=$(awk -v a="$t_start" -v b="${rend[r]}" 'BEGIN {printf "%.1f", b - a}')
    ph=-
    if [[ -n $l && $(field "$l" phase) != - ]]; then ph="$(field "$l" phase)/$(field "$l" phase_name)"; fi
    k2e=$(field "$l" kill_to_exit_ms)
    [[ $k2e != -1.0 ]] || k2e=-
    printf '%-5s %-4s %-5s %-7s %-9s %-11s %-12s %-14s %-5s %s\n' \
        "${rname[r]}" "${rhip[r]}" "${rrc[r]}" "$wall" "$(field "$l" iters)" \
        "$(field "$l" mismatches)" "$(field "$l" reason)" "$ph" "$(field "$l" peer)" "$k2e"
    [[ -n $l || $r -eq $victim ]] ||
        fail "${rname[r]}: no HXLB line in ${rname[r]}.log (exit ${rrc[r]})"
done
if [[ $bench -eq 1 ]]; then
    echo
    grep -h '^HXBENCH' "$LOGDIR/hub.log" 2>/dev/null
fi
echo

expect_rc() {
    [[ ${rrc[$1]} == "$2" ]] || fail "${rname[$1]}: exit ${rrc[$1]}, expected $2"
}
# a survivor is out within limit_ms of the fault
expect_fast() {
    local k
    k=$(field "${hxlb[$1]}" kill_to_exit_ms)
    if [[ $k == - || $k == -1.0 ]]; then
        fail "${rname[$1]}: no kill_to_exit_ms (no fault stamp?)"
    elif ! awk -v k="$k" -v m="$limit_ms" 'BEGIN {exit !(k >= 0 && k < m)}'; then
        fail "${rname[$1]}: ended $k ms after the fault (limit $limit_ms)"
    fi
}
expect_phase() {
    local p
    p=$(field "${hxlb[$1]}" phase)
    [[ " $2 " == *" $p "* ]] || note "note: ${rname[$1]} ended in phase $p, expected $2"
}
# why a survivor ended: peer_death (the pidfd watcher) or abort (device timeout)
expect_reason() {
    local w
    w=$(field "${hxlb[$1]}" reason)
    [[ $w == "$2" ]] || fail "${rname[$1]}: reason $w, expected $2"
}
if [[ $watcher -eq 1 ]]; then want_reason=peer_death; else want_reason=abort; fi

case $kill_test in
    "")
        for ((r = 0; r < nroles; r++)); do expect_rc "$r" 0; done
        [[ $(field "${hxlb[0]}" mismatches) == 0 ]] ||
            fail "hub: mismatches=$(field "${hxlb[0]}" mismatches)"
        ;;
    server)
        expect_rc 0 3
        expect_rc 1 9
        for ((r = 2; r < nroles; r++)); do expect_rc "$r" 3; done
        for ((r = 0; r < nroles; r++)); do
            [[ $r -ne 1 ]] || continue
            expect_fast "$r"
            expect_reason "$r" "$want_reason"
        done
        if [[ $watcher -eq 1 ]]; then expect_phase 0 8; else expect_phase 0 2; fi
        ;;
    hub)
        expect_rc 0 9
        for ((r = 1; r < nroles; r++)); do
            expect_rc "$r" 3
            expect_fast "$r"
            expect_reason "$r" "$want_reason"
            if [[ $watcher -eq 1 ]]; then expect_phase "$r" 8; else expect_phase "$r" "3 4"; fi
        done
        ;;
    hang)
        for ((r = 0; r < nroles; r++)); do expect_rc "$r" 3; done
        expect_reason 0 abort
        expect_phase 0 2
        [[ $(field "${hxlb[0]}" peer) == 0 ]] ||
            fail "hub: the COMBINE abort names peer $(field "${hxlb[0]}" peer), expected 0"
        expect_fast 0
        w=$(field "${hxlb[0]}" waited_ms)
        if [[ ! $w =~ ^[0-9]+$ ]] || ((w * 10 < wait_ms * 8 || w > wait_ms + 3000)); then
            fail "hub: the COMBINE abort waited $w ms, expected about $wait_ms"
        fi
        ;;
esac

if [[ $verdict == FAIL ]]; then
    for m in "${fails[@]}"; do echo "FAIL: $m"; done
fi
echo "$verdict (logs: $LOGDIR)"
[[ $verdict == PASS ]]
