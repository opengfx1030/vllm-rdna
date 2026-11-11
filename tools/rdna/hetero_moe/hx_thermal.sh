#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Junction-temperature logger by PCI bus for the hx soaks on par1-cs25. Box rule: abort a
# sustained run when the 5 s mean of a GPU is >= 90 C or any sample is >= 95 C.
#
#   hx_thermal.sh --out FILE [--buses "d4 d9 dc df e2"] [--kill-pattern ERE]
#                 [--interval SECONDS] [--stop-file FILE] [--wait-pid PID]
#
# One line per sample: "<epoch> <bus>=<C> ...", read from the amdgpu hwmon sensor labelled
# "junction" of /sys/bus/pci/devices/0000:<bus>:00.0 (no rocm-smi, no GPU access). On a
# breach it logs THERMAL ABORT, sends TERM and after 5 s KILL to every process whose
# command line matches --kill-pattern (pkill -f, an extended regex), and exits 9. It
# stops (exit 0) when --stop-file exists or --wait-pid is gone, and then logs one
# THERMAL SUMMARY line per bus (samples, mean, max).
set -uo pipefail
export LC_ALL=C

out=""
buses="d1 d4 d9 dc df e2"
pattern=""
interval=1
stop_file=""
wait_pid=""

die() {
    echo "hx_thermal: $*" >&2
    exit 2
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --out) out=${2:?}; shift 2 ;;
        --buses) buses=${2:?}; shift 2 ;;
        --kill-pattern) pattern=${2:?}; shift 2 ;;
        --interval) interval=${2:?}; shift 2 ;;
        --stop-file) stop_file=${2:?}; shift 2 ;;
        --wait-pid) wait_pid=${2:?}; shift 2 ;;
        -h | --help) sed -n '4,/^set -uo/p' "${BASH_SOURCE[0]}" | grep '^#' | cut -c3-; exit 0 ;;
        *) die "unknown option $1" ;;
    esac
done
[[ -n $out ]] || die "--out is required"
[[ $interval =~ ^[1-9][0-9]*$ ]] || die "--interval must be a positive integer"

declare -A sensor
for b in $buses; do
    [[ $b =~ ^[0-9a-fA-F]{2}$ ]] || die "bad bus '$b'"
    b=${b,,}
    for lbl in /sys/bus/pci/devices/0000:"$b":00.0/hwmon/hwmon*/temp*_label; do
        [[ -r $lbl && $(<"$lbl") == junction ]] && sensor[$b]=${lbl%_label}_input
    done
    [[ -n ${sensor[$b]:-} ]] || die "no junction sensor for bus $b"
done
mkdir -p "$(dirname "$out")"

win=$((5 / interval))
((win >= 1)) || win=1
declare -A hist cnt sum max
for b in "${!sensor[@]}"; do
    hist[$b]=""
    cnt[$b]=0
    sum[$b]=0
    max[$b]=0
done

abort() {
    echo "$(date +%s) THERMAL ABORT bus $1: $2" >>"$out"
    if [[ -n $pattern ]]; then
        pkill -TERM -f -- "$pattern"
        sleep 5
        pkill -KILL -f -- "$pattern"
    fi
    exit 9
}

summary() {
    local b
    for b in $(printf '%s\n' "${!sensor[@]}" | sort); do
        if ((cnt[$b] > 0)); then
            echo "$(date +%s) THERMAL SUMMARY bus $b samples ${cnt[$b]} mean $((sum[$b] / cnt[$b] / 1000)) C max $((max[$b] / 1000)) C" >>"$out"
        fi
    done
}
trap summary EXIT

while :; do
    [[ -z $stop_file || ! -e $stop_file ]] || exit 0
    [[ -z $wait_pid ]] || kill -0 "$wait_pid" 2>/dev/null || exit 0
    line=$(date +%s)
    for b in $(printf '%s\n' "${!sensor[@]}" | sort); do
        v=$(<"${sensor[$b]}") || continue
        [[ $v =~ ^[0-9]+$ ]] || continue
        line+=" $b=$((v / 1000)).$(((v % 1000) / 100))"
        cnt[$b]=$((cnt[$b] + 1))
        sum[$b]=$((sum[$b] + v))
        ((v > max[$b])) && max[$b]=$v
        # shellcheck disable=SC2206 # a space separated list of integers
        h=(${hist[$b]} "$v")
        ((${#h[@]} > win)) && h=("${h[@]: -win}")
        hist[$b]="${h[*]}"
        ((v >= 95000)) && abort "$b" "sample $((v / 1000)) C >= 95 C"
        if ((${#h[@]} == win)); then
            s=0
            for x in "${h[@]}"; do s=$((s + x)); done
            ((s / win >= 90000)) && abort "$b" "5 s mean $((s / win / 1000)) C >= 90 C"
        fi
    done
    echo "$line" >>"$out"
    sleep "$interval"
done
