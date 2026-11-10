#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Runs the hx loopback once per variant of the ordering/visibility switches
# (ctx.flush_mode bits x consumer read mode, see hx_core.cuh) under the per-bus thermal
# logger and prints one summary line per variant (exchanges, mismatches, wall time).
#
#   hx_variants.sh --tag P --bin BIN [--iters N] [--dev-verify] [--timeout SECONDS]
#                  [--hub-hip H] [--srv-hips A,B,C,D] [--extra "OPTS"] [--tool loopback|order] F:M [F:M ...]
#
# F:M = --flush F --rd-mode M of hx_loopback. --dev-verify adds the device-side check
# (no host reference). --extra is appended to every hx_loopback command line. Defaults:
# 1e7 exchanges, hub HIP 1, servers HIP 2,3,4,5, timeout 5400 s per variant. The caller
# holds the leases (run_loopback.sh refuses otherwise). Per variant the logs are under
# ~/w4a8_runs/<P>_f<F>r<M>/; the summary is appended to ~/w4a8_runs/<P>.summary and the
# thermal log is ~/w4a8_runs/<P>.thermal.log, DPM level changes go to <P>.dpm.log. A thermal
# abort ends the chain.
#
# --tool order runs hx_order (the raw ordering race detector) instead: BIN = hx_order,
# --iters = rounds, F:M = --flush F --rd M, hub = consumer (ordinal 0), servers = producers;
# the lease check is done here (the loopback path leaves it to run_loopback.sh).
set -uo pipefail
export LC_ALL=C

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
RUNS=${RUNS:-$HOME/w4a8_runs}
tag=""
bin=""
iters=10000000
dev=0
timeout_s=5400
hub_hip=1
srv_hips=2,3,4,5
extra=""
variants=()
tool=loopback

die() {
    echo "hx_variants: $*" >&2
    exit 2
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --tag) tag=${2:?}; shift 2 ;;
        --bin) bin=${2:?}; shift 2 ;;
        --iters) iters=${2:?}; shift 2 ;;
        --dev-verify) dev=1; shift ;;
        --tool) tool=${2:?}; shift 2 ;;
        --timeout) timeout_s=${2:?}; shift 2 ;;
        --hub-hip) hub_hip=${2:?}; shift 2 ;;
        --srv-hips) srv_hips=${2:?}; shift 2 ;;
        --extra) extra=${2:?}; shift 2 ;;
        -h | --help) sed -n '4,/^set -uo/p' "${BASH_SOURCE[0]}" | grep '^#' | cut -c3-; exit 0 ;;
        [0-9]*:[0-9]*) variants+=("$1"); shift ;;
        *) die "unknown argument $1" ;;
    esac
done
[[ -n $tag && -n $bin ]] || die "--tag and --bin are required"
[[ -x $bin ]] || die "--bin $bin is not executable"
((${#variants[@]})) || die "no variants (F:M ...)"
[[ $tool == loopback || $tool == order ]] || die "--tool must be loopback or order"

# interim box: HIP 0-5 = bus d1 d4 d9 dc df e2
hip_buses=(d1 d4 d9 dc df e2)
buses=""
IFS=, read -r -a hips <<<"$hub_hip,$srv_hips"
for h in "${hips[@]}"; do
    [[ $h =~ ^[0-5]$ ]] || die "HIP '$h' is not one of 0-5"
    buses+=" ${hip_buses[h]}"
done

summary=$RUNS/$tag.summary
thermal=$RUNS/$tag.thermal.log
: >"$summary"
"$HERE/hx_thermal.sh" --out "$thermal" --buses "$buses" --kill-pattern "$bin" --wait-pid $$ &
thermal_pid=$!
python3 "$HERE/hx_dpm_log.py" --out "$RUNS/$tag.dpm.log" --buses "$buses" --wait-pid $$ &
dpm_pid=$!
trap 'kill "$thermal_pid" "$dpm_pid" 2>/dev/null' EXIT

OWNER=${HX_OWNER:-HxServer}
nprod=$((${#hips[@]} - 1))
mode_opts=""
((dev)) && mode_opts="--dev-verify"
for v in "${variants[@]}"; do
    f=${v%%:*}
    m=${v##*:}
    vtag=${tag}_f${f}r${m}
    t0=$(date +%s)
    echo "[hx_variants $(date +%T)] variant flush $f rd-mode $m -> $vtag"
    dv=""
    if [[ $tool == order ]]; then
        for h in "${hips[@]}"; do
            if ((h <= 1)); then lease=a2; else lease=b4; fi
            [[ -e $RUNS/lanes/lease/$OWNER.$lease ]] || die "REFUSED: lease $OWNER.$lease (HIP $h) missing"
        done
        export HIP_VISIBLE_DEVICES=$hub_hip,$srv_hips
        # shellcheck disable=SC2086 # extra is an option list
        timeout --signal=TERM --kill-after=20 "$timeout_s" "$bin" --nprod "$nprod" --rounds "$iters" \
            --flush "$f" --rd "$m" --report-s 30 $extra >"$RUNS/$vtag.out" 2>&1
        rc=$?
        unset HIP_VISIBLE_DEVICES
        row=$(grep -h "ORDER done" "$RUNS/$vtag.out" | tail -n 1)
        [[ -n $row ]] || row=$(grep -h "ORDER rounds=\|ORDER abort" "$RUNS/$vtag.out" | tail -n 1)
        row=${row:-no-order-line}
    else
        # shellcheck disable=SC2086 # mode_opts and extra are option lists
        "$HERE/run_loopback.sh" --tag "$vtag" --hub-hip "$hub_hip" --srv-hips "$srv_hips" --bin "$bin" \
            --timeout "$timeout_s" -- $mode_opts --iters "$iters" --flush "$f" --rd-mode "$m" \
            --no-stop-on-mismatch --log-every 1000000 $extra >"$RUNS/$vtag.out" 2>&1
        rc=$?
        hub_log=$RUNS/$vtag/hxlb/soak/hub.log
        row=$(awk '$1 == "hub" && NF >= 6 {print "rc=" $3 " iters=" $5 " mismatches=" $6}' "$RUNS/$vtag.out" | tail -n 1)
        dv=$(grep -h "DEVVERIFY" "$hub_log" 2>/dev/null | tail -n 1)
        row=${row:-no-hub-row}
    fi
    line="flush=$f rd=$m run_rc=$rc $row wall_s=$(($(date +%s) - t0)) ${dv:+| $dv}"
    echo "$line" | tee -a "$summary"
    if grep -q "THERMAL ABORT" "$thermal" 2>/dev/null; then
        echo "THERMAL ABORT: chain stopped" | tee -a "$summary"
        exit 9
    fi
done
