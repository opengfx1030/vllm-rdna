#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Runs the hx loopback once per variant of the ordering/visibility switches
# (ctx.flush_mode bits x consumer read mode, see hx_core.cuh) under the per-bus thermal
# logger and prints one summary line per variant (exchanges, mismatches, wall time).
#
#   hx_variants.sh --tag P --bin BIN [--iters N] [--dev-verify] [--timeout SECONDS]
#                  [--hub-hip H] [--srv-hips A,B,C,D] [--extra "OPTS"] [--tool loopback|order]
#                  [--noise S,D[:MODE]] [--noise-bin PATH] F:M [F:M ...]
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
#
# --noise S,D[:MODE] runs hx_noise (MODE kernel|copy|read|bidir-kernel, default kernel;
# source GPU S, destination GPU D as HIP numbers, the leases of both are checked) next to
# every variant and stops it afterwards: PCIe background traffic as a race amplifier.
# Useful pairs: 0,1 (into the hub), 2,1 (a server floods the hub on the path of its own
# replies), 1,2 (the hub floods a server on the request path). --noise-args "OPTS" adds
# hx_noise options (for example "--bytes 4194304 --idle-us 500" for bursts).
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
noise=""
noise_bin=$RUNS/hxnoise/hx_noise
noise_pid=""
noise_args=""

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
        --noise) noise=${2:?}; shift 2 ;;
        --noise-bin) noise_bin=${2:?}; shift 2 ;;
        --noise-args) noise_args=${2:?}; shift 2 ;;
        -h | --help) sed -n '4,/^set -uo/p' "${BASH_SOURCE[0]}" | grep '^#' | cut -c3-; exit 0 ;;
        [0-9]*:[0-9]*) variants+=("$1"); shift ;;
        *) die "unknown argument $1" ;;
    esac
done
[[ -n $tag && -n $bin ]] || die "--tag and --bin are required"
[[ -x $bin ]] || die "--bin $bin is not executable"
((${#variants[@]})) || die "no variants (F:M ...)"
[[ $tool == loopback || $tool == order ]] || die "--tool must be loopback or order"
n_src=""
n_dst=""
n_mode=kernel
if [[ -n $noise ]]; then
    n_src=${noise%%,*}
    n_rest=${noise#*,}
    n_dst=${n_rest%%:*}
    [[ $n_rest == *:* ]] && n_mode=${n_rest#*:}
    [[ $n_src =~ ^[0-5]$ && $n_dst =~ ^[0-5]$ && $n_src != "$n_dst" ]] ||
        die "--noise wants S,D[:MODE] with two different HIPs 0-5"
    [[ -x $noise_bin ]] || die "--noise-bin $noise_bin is not executable"
fi

# interim box: HIP 0-5 = bus d1 d4 d9 dc df e2
hip_buses=(d1 d4 d9 dc df e2)
buses=""
IFS=, read -r -a hips <<<"$hub_hip,$srv_hips"
for h in "${hips[@]}"; do
    [[ $h =~ ^[0-5]$ ]] || die "HIP '$h' is not one of 0-5"
    buses+=" ${hip_buses[h]}"
done
if [[ -n $noise ]]; then buses+=" ${hip_buses[n_src]} ${hip_buses[n_dst]}"; fi
# shellcheck disable=SC2086 # word splitting of the bus list is intended
buses=$(printf '%s\n' $buses | sort -u | tr '\n' ' ')

summary=$RUNS/$tag.summary
thermal=$RUNS/$tag.thermal.log
: >"$summary"
"$HERE/hx_thermal.sh" --out "$thermal" --buses "$buses" --kill-pattern "$bin${noise:+|$noise_bin}" --wait-pid $$ &
thermal_pid=$!
python3 "$HERE/hx_dpm_log.py" --out "$RUNS/$tag.dpm.log" --buses "$buses" --wait-pid $$ &
dpm_pid=$!
trap 'kill "$thermal_pid" "$dpm_pid" ${noise_pid:+"$noise_pid"} 2>/dev/null' EXIT

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
    nz=""
    if [[ -n $noise ]]; then
        for h in "$n_src" "$n_dst"; do
            if ((h <= 1)); then lease=a2; else lease=b4; fi
            [[ -e $RUNS/lanes/lease/$OWNER.$lease ]] || die "REFUSED: lease $OWNER.$lease (noise HIP $h) missing"
        done
        # shellcheck disable=SC2086 # noise_args is an option list
        env HIP_VISIBLE_DEVICES="$n_src,$n_dst" \
            "$noise_bin" --mode "$n_mode" --seconds 0 --report-s 30 $noise_args >"$RUNS/$vtag.noise.log" 2>&1 &
        noise_pid=$!
        sleep 5
    fi
    if [[ $tool == order ]]; then
        for h in "${hips[@]}"; do
            if ((h <= 1)); then lease=a2; else lease=b4; fi
            [[ -e $RUNS/lanes/lease/$OWNER.$lease ]] || die "REFUSED: lease $OWNER.$lease (HIP $h) missing"
        done
        # shellcheck disable=SC2086 # extra is an option list
        env HIP_VISIBLE_DEVICES="$hub_hip,$srv_hips" \
            timeout --signal=TERM --kill-after=20 "$timeout_s" "$bin" --nprod "$nprod" --rounds "$iters" \
            --flush "$f" --rd "$m" --report-s 30 $extra >"$RUNS/$vtag.out" 2>&1
        rc=$?
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
    if [[ -n $noise_pid ]]; then
        kill -TERM "$noise_pid" 2>/dev/null
        wait "$noise_pid" 2>/dev/null
        noise_pid=""
        nz=$(grep -h "NOISE mode=" "$RUNS/$vtag.noise.log" | tail -n 1)
    fi
    line="flush=$f rd=$m run_rc=$rc $row wall_s=$(($(date +%s) - t0)) ${dv:+| $dv} ${nz:+| $nz}"
    echo "$line" | tee -a "$summary"
    if grep -q "THERMAL ABORT" "$thermal" 2>/dev/null; then
        echo "THERMAL ABORT: chain stopped" | tee -a "$summary"
        exit 9
    fi
done
