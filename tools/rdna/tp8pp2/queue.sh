#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Arm queue: waits until no other lease holder is on the slots, takes the leases, runs
# driver.sh once per line of QUEUE_FILE (in order), releases the leases when the queue
# is empty or at a boot boundary when asked.
#
#   OWNER=<lease holder> [LEASE_SLOTS=AB] bash queue.sh QUEUE_FILE
#   (from a fresh ssh login:  setsid nohup env OWNER=... bash queue.sh Q > q.log 2>&1 < /dev/null &)
#
# QUEUE_FILE: one arm per line (blank lines and lines starting with # are skipped):
#
#   SHA=<40 hex> CFG=tp8 ARM=tp8-s1p HOOK=1 CONC_REPEAT=10 CELLS_SPEC=none QUALITY=0
#
# The words are KEY=value assignments for `env ... bash driver.sh` (see driver.sh; a
# value with spaces is quoted: SERVE_ENV="A=1 B=2"). The line is run by `bash -c`, so
# it is shell text: only queue files you wrote yourself.
#
# Progress: $RUN_DIR/<QUEUE_FILE name>.next holds the number of arms already run, so a
# stopped queue resumes where it left off. Control (checked before every arm):
#   touch $RUN_DIR/queue.pause   finish the current arm, release the leases, exit
# One status line per arm goes to $RUN_DIR/queue.status.
set -uo pipefail
# shellcheck source=slots.sh
source "$(dirname "${BASH_SOURCE[0]}")/slots.sh"
OWNER=${OWNER:?OWNER=<lease holder name>}
export OWNER LANES BOOT_LOCK RUN_DIR="$R"
QFILE=${1:?QUEUE_FILE}
SLOTS=${LEASE_SLOTS:-AB}
POLL_S=${POLL_S:-30}
PAUSE=$R/queue.pause
NEXT=$R/$(basename "$QFILE").next
mkdir -p "$R/arms" "$LEASE"

say() { echo "[$(date '+%F %T')] $*"; }
foreign_holders() {
    local x f
    for x in A B; do
        [[ $SLOTS == *$x* ]] || continue
        for f in "$LEASE"/*.slot"$x"*; do
            [[ -e $f ]] || continue
            [[ ${f##*/} == "$OWNER".* ]] || echo "${f##*/}"
        done
    done
}
release() {
    rm -f "$LEASE/$OWNER.slotA" "$LEASE/$OWNER.slotB"
}
# shellcheck disable=SC2329  # installed with trap
on_exit() {
    release
    say "queue exit: leases released"
}
acquire() {
    local x
    until [[ -z $(foreign_holders) ]]; do
        if [[ -e $PAUSE ]]; then
            return 1
        fi
        say "waiting for lease holders: $(foreign_holders | tr '\n' ' ')"
        sleep "$POLL_S"
    done
    for x in A B; do
        [[ $SLOTS == *$x* ]] && echo "$(date -Is) queue $(basename "$QFILE")" > "$LEASE/$OWNER.slot$x"
    done
    sleep 5
    if [[ -n $(foreign_holders) ]]; then
        say "lease conflict after taking the files: backing off"
        release
        return 1
    fi
    say "leases taken: slots $SLOTS"
}

trap on_exit EXIT
trap 'say "signal: leaving"; exit 143' TERM INT HUP
idx=0
start=$(cat "$NEXT" 2> /dev/null || echo 0)
# shellcheck disable=SC2094  # QFILE is only read; NEXT and the status log are other files
while IFS= read -r line || [[ -n $line ]]; do
    [[ -z ${line//[[:space:]]/} || $line == \#* ]] && continue
    idx=$((idx + 1))
    ((idx <= start)) && continue
    if [[ -e $PAUSE ]]; then
        say "$PAUSE present: stopping before arm $idx"
        break
    fi
    until [[ -e $LEASE/$OWNER.slotA || -e $LEASE/$OWNER.slotB ]] && [[ -z $(foreign_holders) ]]; do
        acquire || {
            [[ -e $PAUSE ]] && break 2
            sleep "$POLL_S"
        }
    done
    say "arm $idx: $line"
    t0=$(date +%s)
    bash -c "env $line bash '$H/driver.sh'"
    rc=$?
    echo "$(date '+%F %T') queue $(basename "$QFILE") arm $idx rc=$rc $((($(date +%s) - t0) / 60)) min: $line" >> "$R/queue.status"
    echo "$idx" > "$NEXT"
done < "$QFILE"
say "queue done (arms run so far: $(cat "$NEXT" 2> /dev/null || echo 0))"
