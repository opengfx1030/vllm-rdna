#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Evidence run for the KFD queue evictions of the pinned host memory. It needs no
# GPU and starts no server: it starts, as root (`sudo -n`), pinned_sampler.py (every
# VLLM::Worker on the host, 1 Hz) and the raw KFD event reader, writes what decides
# reclaim (sysctls, zone watermarks, swap and zram, THP, KSM, MGLRU, proactive reclaim,
# RLIMIT_MEMLOCK and cgroup limits of the workers) at the start and at the end, and
# turns the samples into report.md (pinned_report.py).
#
#   pinned_evidence.sh start  [OUT]   env: HOURS=1 PY KFD_RAW SAMPLER_ARGS SUDO
#   pinned_evidence.sh status OUT
#   pinned_evidence.sh stop   OUT     SIGTERM both, end snapshot, report.md
#   pinned_evidence.sh report OUT     report.md again (the sampler may still run); env: SERVE_LOGS
#
# OUT defaults to ~/w4a8_runs/evictfix/<timestamp>. Both children are started with
# setsid nohup, stopped with SIGTERM only, and read only /proc, /sys and
# /sys/kernel/debug/kfd/mem_limit; nothing here reads amdgpu_vm_info (minutes).
# KFD_RAW is the command of the raw reader, which is called with <log> <hours>;
# empty skips it (the report then uses the evicted_ms jumps alone). SERVE_LOGS lists
# vLLM server logs (space separated) whose "(Worker_TP0_EP0 pid=N)" lines give the
# report the rank of workers that were seen before they set their title.
set -uo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
VENV=${VENV:-$HOME/Apps/vllm/venv-7.14.0_0.32.0}
PY=${PY:-$VENV/bin/python}
HOURS=${HOURS:-1}
KFD_RAW=${KFD_RAW-python3 $HOME/w4a8_runs/rdna_ar_wedge/kfd_smi_raw.py}
SAMPLER_ARGS=${SAMPLER_ARGS:-}
SERVE_LOGS=${SERVE_LOGS:-}
SUDO=${SUDO:-sudo -n}
VM_KNOBS="swappiness min_free_kbytes watermark_scale_factor watermark_boost_factor
    compaction_proactiveness compact_unevictable_allowed vfs_cache_pressure
    page-cluster dirty_ratio dirty_background_ratio overcommit_memory
    zone_reclaim_mode"

say() { printf '[%s] %s\n' "$(date '+%F %T')" "$*"; }
die() {
    say "ABORT: $*"
    exit 2
}

snapshot() { # snapshot FILE: the host state that decides reclaim
    local f=$1 k z p cg w
    {
        echo "# $(date -Is) kernel $(uname -r)"
        uptime
        echo "## meminfo"
        grep -E '^(MemTotal|MemFree|MemAvailable|Cached|Shmem|Mlocked|Unevictable|SwapTotal|SwapFree|AnonPages|Dirty|Writeback):' /proc/meminfo
        echo "## vm sysctls"
        for k in $VM_KNOBS; do
            echo "vm.$k=$(cat "/proc/sys/vm/$k" 2> /dev/null || echo n/a)"
        done
        echo "kernel.numa_balancing=$(cat /proc/sys/kernel/numa_balancing 2> /dev/null || echo n/a)"
        echo "## zone watermarks (pages)"
        awk '/^Node/ {z=$0} /^ +(pages free|boost|min|low|high|managed) / {print z ":" $0}' /proc/zoneinfo
        echo "## swap"
        cat /proc/swaps
        for z in /sys/block/zram*; do
            [[ -d $z ]] || continue
            echo "$z: disksize $(cat "$z/disksize") algo $(cat "$z/comp_algorithm") mm_stat $(cat "$z/mm_stat")"
        done
        echo "## memory features"
        for k in /sys/kernel/mm/transparent_hugepage/enabled /sys/kernel/mm/transparent_hugepage/defrag \
            /sys/kernel/mm/transparent_hugepage/khugepaged/defrag /sys/kernel/mm/ksm/run \
            /sys/kernel/mm/lru_gen/enabled /sys/kernel/mm/lru_gen/min_ttl_ms \
            /sys/module/damon_reclaim/parameters/enabled; do
            [[ -r $k ]] && echo "$k: $(cat "$k")"
        done
        echo "## pressure"
        cat /proc/pressure/memory /proc/pressure/io 2> /dev/null
        echo "## amdkfd parameters"
        grep -r . /sys/module/amdkfd/parameters 2> /dev/null
        echo "## KFD system memory"
        # shellcheck disable=SC2086
        timeout 10 $SUDO cat /sys/kernel/debug/kfd/mem_limit 2> /dev/null || echo "mem_limit: n/a"
        echo "## login shell: ulimit -l $(ulimit -l)"
        echo "## workers: RLIMIT_MEMLOCK, cgroup memory"
        for p in $(pgrep -f 'VLLM::Worker' 2> /dev/null); do
            w=$(tr '\0' ' ' < "/proc/$p/cmdline" 2> /dev/null | cut -c1-48)
            echo "pid $p $w"
            grep -E 'Max locked memory' "/proc/$p/limits" 2> /dev/null
            cg=$(sed -n 's/^0:://p' "/proc/$p/cgroup" 2> /dev/null)
            for k in memory.events memory.high memory.max memory.swap.max memory.current; do
                [[ -r /sys/fs/cgroup$cg/$k ]] && echo "  $cg/$k: $(tr '\n' ' ' < "/sys/fs/cgroup$cg/$k")"
            done
        done
        echo "## vmstat"
        cat /proc/vmstat
    } > "$f" 2>&1
}

start_child() { # start_child NAME LOG cmd... : root, own session, pid file in $OUT
    local name=$1 log=$2
    shift 2
    # shellcheck disable=SC2086,SC2016
    $SUDO setsid nohup sh -c 'echo $$ > "$1"; shift; exec "$@"' sh "$OUT/$name.pid" "$@" > "$log" 2>&1 &
}

rkill() { # rkill SIGNAL PID: as root, with the shell's kill (no /usr/bin/kill needed)
    # shellcheck disable=SC2086,SC2016
    $SUDO sh -c 'kill "-$1" "$2"' sh "$1" "$2" 2> /dev/null
}

alive() { # alive NAME
    local pid
    pid=$(cat "$OUT/$1.pid" 2> /dev/null) || return 1
    rkill 0 "$pid"
}

cmd_report() {
    local args=() log
    [[ -s $OUT/kfd_events.log ]] && args=(--kfd-events "$OUT/kfd_events.log")
    for log in $SERVE_LOGS; do
        args+=(--serve-log "$log")
    done
    "$PY" "$HERE/pinned_report.py" "$OUT" ${args[@]+"${args[@]}"} --md "$OUT/report.md" || return 1
    say "report: $OUT/report.md"
}

cmd_start() {
    [[ -e $OUT/START ]] && die "$OUT was started already"
    # shellcheck disable=SC2086
    $SUDO true 2> /dev/null || die "passwordless sudo is needed (pagemap PFNs are root-only)"
    "$PY" -c 'import numpy' 2> /dev/null || die "$PY has no numpy (the sampler needs it)"
    mkdir -p "$OUT" || die "cannot create $OUT"
    snapshot "$OUT/host_start.txt"
    # shellcheck disable=SC2086
    start_child sampler "$OUT/sampler.out" "$PY" "$HERE/pinned_sampler.py" \
        --out "$OUT" --hours "$HOURS" $SAMPLER_ARGS
    if [[ -n $KFD_RAW ]]; then
        # shellcheck disable=SC2086
        start_child kfd_raw "$OUT/kfd_raw.out" $KFD_RAW "$OUT/kfd_events.log" "$HOURS"
    fi
    sleep 3
    date -Is > "$OUT/START"
    alive sampler || die "the sampler is not running: see $OUT/sampler.out"
    if [[ -n $KFD_RAW ]] && ! alive kfd_raw; then
        say "WARNING: the raw KFD reader is not running (see $OUT/kfd_raw.out); the report will use evicted_ms jumps alone"
    fi
    say "sampling for ${HOURS} h into $OUT; stop with: $0 stop $OUT"
}

cmd_status() {
    local n f age
    for n in sampler kfd_raw; do
        if [[ ! -e $OUT/$n.pid ]]; then
            echo "$n: not started"
        elif alive "$n"; then
            echo "$n: running (pid $(cat "$OUT/$n.pid"))"
        else
            echo "$n: not running"
        fi
    done
    f=$OUT/samples.jsonl
    if [[ -s $f ]]; then
        age=$(($(date +%s) - $(stat -c %Y "$f")))
        echo "samples: $(wc -l < "$f") lines, the last $age s ago; started $(cat "$OUT/START" 2> /dev/null)"
    else
        echo "samples: none yet"
    fi
}

cmd_stop() {
    local n pid
    for n in sampler kfd_raw; do
        pid=$(cat "$OUT/$n.pid" 2> /dev/null) || continue
        rkill TERM "$pid" || true
    done
    for _ in $(seq 1 30); do
        alive sampler || break
        sleep 1
    done
    alive sampler && say "WARNING: the sampler is still running after 30 s; it was only sent SIGTERM"
    snapshot "$OUT/host_end.txt"
    cmd_report
}

[[ $# -ge 1 ]] || die "usage: $0 start [OUT] | status OUT | stop OUT | report OUT"
cmd=$1
case $cmd in
    start) OUT=${2:-$HOME/w4a8_runs/evictfix/$(date +%Y%m%d-%H%M%S)} ;;
    status | stop | report) OUT=${2:?usage: $0 $cmd OUT} ;;
    *) die "unknown command $cmd" ;;
esac
"cmd_$cmd"
