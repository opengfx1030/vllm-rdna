# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Lease helpers of the Disagg drivers (~/w4a8_runs/lanes/lease/README.md). Source after
# slots.sh. Needs $VENV and a writable $R (scratch for the memory probe's stderr).
#   take_leases SLOT...   request every slot, wait until no other holder file exists for any
#                         of them and every GPU of every slot has >= 28 GiB free, then write
#                         the holder files
#   release_slot SLOT     remove my holder file and my request for one slot
#   release_all           release every slot taken here and remove my quiet file
LEASE=${LEASE:-$HOME/w4a8_runs/lanes/lease}
QUIET=${QUIET:-$HOME/w4a8_runs/lanes/quiet}
OWNER=${OWNER:-Disagg}
POLL_S=${POLL_S:-20}
LEASE_MINUTES=${LEASE_MINUTES:-240}
LEASE_WAIT_MAX_S=${LEASE_WAIT_MAX_S:-43200}
LEASE_HELD_SLOTS=()
LEASE_REQ_SLOTS=()

say() { echo "[$(date '+%F %T')] $*"; }
# shellcheck disable=SC2034 # STATUS is read by the sourcing driver
die() { say "FATAL: $*"; STATUS=1; exit 1; }
iso_in() { date -Is -d "+$1 minutes"; }
lease_req() { echo "$LEASE/slot$1.request"; }
lease_hold() { echo "$LEASE/$OWNER.slot$1"; }
req_is_mine() { grep -q "^$OWNER " "$(lease_req "$1")" 2> /dev/null; }
others_hold() {
    local s=$1 f
    for f in "$LEASE"/*.slot"$s"; do
        [[ -e $f && $f != "$(lease_hold "$s")" ]] && return 0
    done
    return 1
}
lease_ls() {
    local f
    for f in "$LEASE"/*; do printf '%s ' "${f##*/}"; done
}
free_gib() {
    HIP_VISIBLE_DEVICES=$(slot_hip "$1") "$VENV/bin/python" -c '
import torch
print(" ".join(f"{torch.cuda.mem_get_info(i)[0] / 2**30:.1f}" for i in range(torch.cuda.device_count())))' 2> "$R/free_gib.err"
}
enough_memory() { # enough_memory SLOT: >= 28 GiB free on each of the slot's 4 GPUs
    local f g n=0
    f=$(free_gib "$1") || { say "free_gib failed: $(tail -2 "$R/free_gib.err" | tr '\n' ' ')"; return 1; }
    say "free GiB on slot $1 (HIP $(slot_hip "$1")): $f"
    for g in $f; do
        n=$((n + 1))
        awk -v g="$g" 'BEGIN { exit !(g >= 28) }' || return 1
    done
    ((n == 4))
}
take_leases() {
    local t0 s ok end
    t0=$(date +%s)
    for s in "$@"; do
        until ( set -C; echo "$OWNER $(date -Is) disagg g0 slot $s" > "$(lease_req "$s")" ) 2> /dev/null || req_is_mine "$s"; do
            say "waiting: foreign request on slot $s: $(cat "$(lease_req "$s")" 2> /dev/null)"
            (($(date +%s) - t0 > LEASE_WAIT_MAX_S)) && die "no lease within ${LEASE_WAIT_MAX_S}s: foreign request on slot $s"
            sleep "$POLL_S"
        done
        LEASE_REQ_SLOTS+=("$s")
        say "request slot $s: $(cat "$(lease_req "$s")")"
    done
    while :; do
        ok=1
        for s in "$@"; do others_hold "$s" && ok=0; done
        if ((ok)); then
            for s in "$@"; do enough_memory "$s" || ok=0; done
        fi
        ((ok)) && break
        say "waiting: lease dir: $(lease_ls)"
        (($(date +%s) - t0 > LEASE_WAIT_MAX_S)) && die "no lease within ${LEASE_WAIT_MAX_S}s: holders or memory"
        sleep "$POLL_S"
    done
    end=$(iso_in "$LEASE_MINUTES")
    for s in "$@"; do
        echo "$OWNER $end disagg g0 slot $s HIP $(slot_hip "$s")" > "$(lease_hold "$s")"
        LEASE_HELD_SLOTS+=("$s")
    done
    say "lease taken until $end: $(lease_ls)"
}
release_slot() {
    rm -f "$(lease_hold "$1")"
    if req_is_mine "$1"; then rm -f "$(lease_req "$1")"; fi
    say "slot $1 released (holder file and request removed)"
}
release_all() {
    local s
    for s in "${LEASE_HELD_SLOTS[@]}" "${LEASE_REQ_SLOTS[@]}"; do
        [[ -e $(lease_hold "$s") ]] || req_is_mine "$s" && release_slot "$s"
    done
    rm -f "$QUIET/$OWNER"
    say "lease released (quiet file removed)"
}
