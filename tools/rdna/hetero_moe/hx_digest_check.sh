#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Validation of the digest check (ctx.flush_mode bit HX_DIGEST, csrc/rocm/hetero_moe/hx_core.cuh) on
# the hx loopback, one line per case: a clean run and positive controls with a 256 B chunk that
# lands late (after its flag), each with its expected outcome, then PASS or FAIL.
#
#   hx_digest_check.sh --bin BIN --fi-bin BIN_FI [--tag T] [--hub-hip H] [--srv-hips A,B,..]
#                      [--iters N] [--cases c1,c2,..] [--srv S] [--chunk C] [--late-us U]
#
# BIN is a normal hx_loopback build, BIN_FI one built with -DHX_FAULT_INJECT
# (HX_LB_CXXFLAGS=-DHX_FAULT_INJECT run_loopback.sh --build-only --tag T). The caller holds the
# leases of every GPU used (run_loopback.sh refuses otherwise). Defaults: hub HIP 1, servers
# HIP 2,3,4,5, 200000 exchanges for the clean case, 3000 for the controls, injection at exchange
# 1000 into server 1 (server 0 with one server), chunk 3, 20000 us for the "late" cases.
#
# Cases (default: all), all with --dev-verify --digest:
#   clean     --T-list 3,1,6,24,48: every process DIGSTAT bad=0 unhealed=0 ok>0, 0 mismatches, exit 0
#   rep_heal  the reply chunk of server S lands 100 us after its flag: the hub counts bad=1
#             healed=1, the combine output is right (0 mismatches), nobody aborts
#   rep_late  the same 20 ms late (--late-us): hub bad=1 unhealed=1, abort record DIGEST, exit 3
#   req_heal  the request chunk to server S lands 100 us late: that server counts bad=1 healed=1
#   req_late  the same 20 ms late: server S aborts with DIGEST
set -uo pipefail
export LC_ALL=C

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
RUNS=${RUNS:-$HOME/w4a8_runs}
tag=hxdig
bin=""
fibin=""
hub_hip=1
srv_hips=2,3,4,5
iters=200000
cases=clean,rep_heal,rep_late,req_heal,req_late
srv=""
chunk=3
late_us=20000
inj_at=1000

die() {
    echo "hx_digest_check: $*" >&2
    exit 2
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --bin) bin=${2:?}; shift 2 ;;
        --fi-bin) fibin=${2:?}; shift 2 ;;
        --tag) tag=${2:?}; shift 2 ;;
        --hub-hip) hub_hip=${2:?}; shift 2 ;;
        --srv-hips) srv_hips=${2:?}; shift 2 ;;
        --iters) iters=${2:?}; shift 2 ;;
        --cases) cases=${2:?}; shift 2 ;;
        --srv) srv=${2:?}; shift 2 ;;
        --chunk) chunk=${2:?}; shift 2 ;;
        --late-us) late_us=${2:?}; shift 2 ;;
        -h | --help) sed -n '4,/^set -uo/p' "${BASH_SOURCE[0]}" | grep '^#' | cut -c3-; exit 0 ;;
        *) die "unknown option $1" ;;
    esac
done
[[ -x $bin ]] || die "--bin BIN (an executable hx_loopback) is required"
IFS=, read -ra hips <<<"$srv_hips"
nsrv=${#hips[@]}
if [[ -z $srv ]]; then srv=$((nsrv > 1 ? 1 : 0)); fi
((srv < nsrv)) || die "--srv $srv is not below the number of servers ($nsrv)"

# thermal guard of the whole script (box rule: 5 s mean >= 90 C or a sample >= 95 C aborts)
buses=""
bus_of=(d1 d4 d9 dc df e2)
for h in "$hub_hip" "${hips[@]}"; do
    [[ $h =~ ^[0-5]$ ]] || die "HIP $h is not one of 0-5"
    b=${bus_of[$h]}
    [[ " $buses " == *" $b "* ]] || buses="$buses $b"
done
thermal=$RUNS/${tag}.thermal.log
"$HERE/hx_thermal.sh" --out "$thermal" --buses "${buses# }" --kill-pattern "hx_loopback" --wait-pid $$ &

verdict=PASS
fails=()
fail() {
    verdict=FAIL
    fails+=("$*")
}

# dig LOG KEY -> KEY= of the last DIGSTAT line of LOG ("-" when none)
dig() {
    local l
    l=$(grep -h -o 'DIGSTAT .*' "$1" 2>/dev/null | tail -n 1)
    if [[ $l =~ (^|[[:space:]])$2=([^[:space:]]*) ]]; then echo "${BASH_REMATCH[2]}"; else echo -; fi
}
# hxlb LOG KEY -> KEY= of the last HXLB line of LOG
hxlb() {
    local l
    l=$(grep -h '^HXLB ' "$1" 2>/dev/null | tail -n 1)
    if [[ $l =~ (^|[[:space:]])$2=([^[:space:]]*) ]]; then echo "${BASH_REMATCH[2]}"; else echo -; fi
}
# expect CASE WHAT GOT WANT
expect() {
    [[ $3 == "$4" ]] || fail "$1: $2 is $3, expected $4"
}

# run NAME BINARY ARGS...: one run_loopback, logs in $RUNS/<tag>_<name>/hxlb/dig
run() {
    local name=$1 b=$2
    shift 2
    echo "[digest-check] case $name: $*"
    "$HERE/run_loopback.sh" --tag "${tag}_$name" --name dig --hub-hip "$hub_hip" --srv-hips "$srv_hips" \
        --bin "$b" --timeout 900 -- --dev-verify --digest "$@" >"$RUNS/${tag}_$name.out" 2>&1
    echo "[digest-check]   run_loopback rc=$?"
}
logdir() { echo "$RUNS/${tag}_$1/hxlb/dig"; }

roles=(hub)
for ((i = 0; i < nsrv; i++)); do roles+=("srv$i"); done

IFS=, read -ra todo <<<"$cases"
for c in "${todo[@]}"; do
    case $c in
        clean)
            run clean "$bin" --T-list 3,1,6,24,48 --iters "$iters"
            d=$(logdir clean)
            for r in "${roles[@]}"; do
                expect clean "$r bad" "$(dig "$d/$r.log" bad)" 0
                expect clean "$r unhealed" "$(dig "$d/$r.log" unhealed)" 0
                [[ $(dig "$d/$r.log" ok) =~ ^[1-9][0-9]*$ ]] || fail "clean: $r counted no checked exchange"
                expect clean "$r exit" "$(hxlb "$d/$r.log" exit)" 0
                expect clean "$r mismatches" "$(hxlb "$d/$r.log" mismatches)" 0
            done
            ;;
        rep_heal | rep_late | req_heal | req_late)
            [[ -x $fibin ]] || die "$c needs --fi-bin (a build with -DHX_FAULT_INJECT)"
            us=100
            [[ $c == *_late ]] && us=$late_us
            req=()
            [[ $c == req_* ]] && req=(--inject-late-req)
            run "$c" "$fibin" --T-list 3 --iters 3000 --inject-late "$inj_at" --inject-late-us "$us" \
                --inject-srv "$srv" --inject-chunk "$chunk" "${req[@]}"
            d=$(logdir "$c")
            who=hub
            [[ $c == req_* ]] && who=srv$srv
            if [[ $c == *_heal ]]; then
                expect "$c" "$who bad" "$(dig "$d/$who.log" bad)" 1
                expect "$c" "$who healed" "$(dig "$d/$who.log" healed)" 1
                expect "$c" "$who unhealed" "$(dig "$d/$who.log" unhealed)" 0
                for r in "${roles[@]}"; do
                    expect "$c" "$r exit" "$(hxlb "$d/$r.log" exit)" 0
                    expect "$c" "$r mismatches" "$(hxlb "$d/$r.log" mismatches)" 0
                done
            else
                expect "$c" "$who unhealed" "$(dig "$d/$who.log" unhealed)" 1
                expect "$c" "$who exit" "$(hxlb "$d/$who.log" exit)" 3
                expect "$c" "$who phase" "$(hxlb "$d/$who.log" phase_name)" DIGEST
            fi
            ;;
        *) die "unknown case $c" ;;
    esac
    echo "[digest-check] case $c done"
    for r in "${roles[@]}"; do
        grep -h -o 'DIGSTAT .*' "$(logdir "$c")/$r.log" 2>/dev/null | tail -n 1 | cut -c1-210
    done
done

echo
if [[ $verdict == FAIL ]]; then
    for m in "${fails[@]}"; do echo "FAIL: $m"; done
fi
echo "hx_digest_check: $verdict"
[[ $verdict == PASS ]]
