#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Shared GPU-lease carrier for the Inkling-2 HI2 kit (AGENTS.md, "Two-slot box
# operation"; ~/w4a8_runs/lanes/lease/README.md):
#
#   holder files <owner>.<slot>; request files <slot>.request (one per slot, per
#   the lane README).  The next job queues behind another owner's request - it is
#   never overwritten - drops its own holder when the recorded pid is gone, arms
#   the release trap before writing, and takes the lease only when no other holder
#   exists and every GPU of the lease has >= 28 GiB free (the gate is re-checked
#   even when reusing its own holder); the holder is removed by the release trap.
#
# Expects OWNER, LEASE_DIR, SLOT and (for the free-VRAM gate) EXPECT_PCI_BUSES as
# a comma list of PCI bus numbers.  Reads DRM_SYSFS (default /sys/class/drm; the
# sysfs mem_info_vram_* files only - no GPU is opened), LEASE_FREE_GIB (28),
# LEASE_POLL_SECONDS (30), LEASE_WAIT_SECONDS (1800).
#
# Source it; do not execute.  serve.sh owns the acquire/release lifecycle and
# re-arms its combined EXIT/TERM trap; smoke.sh only refuses a foreign holder.
# shellcheck shell=bash

DRM_SYSFS=${DRM_SYSFS:-/sys/class/drm}
LEASE_FREE_GIB=${LEASE_FREE_GIB:-28}
LEASE_POLL_SECONDS=${LEASE_POLL_SECONDS:-30}
LEASE_WAIT_SECONDS=${LEASE_WAIT_SECONDS:-1800}
LEASE_REQUEST=${LEASE_REQUEST:-}
LEASE_HOLDER_CREATED=${LEASE_HOLDER_CREATED:-}
# Inherit mode: a parent that already holds this slot's lease (the one-shot
# driver) lets its child boots skip request/holder creation and release.  The
# holder must be ours, its pid alive, and the free gate recorded by the owner
# at acquisition time must be >= LEASE_FREE_GIB (a queued request does not
# interrupt the holder's own nested boots; the requester waits for our release).
LEASE_INHERIT=${LEASE_INHERIT:-0}
LEASE_INHERIT_ACTIVE=${LEASE_INHERIT_ACTIVE:-0}

lease_holders() {
    local f
    for f in "$LEASE_DIR"/*."$SLOT"; do
        [[ -e $f ]] || continue
        printf '%s\n' "$f"
    done
}

# Print the first holder that is not ours; return 1 when there is none.
lease_foreign_holder() {
    local f
    while IFS= read -r f; do
        [[ -z $f ]] && continue
        [[ $(basename "$f") == "$OWNER.$SLOT" ]] && continue
        printf '%s\n' "$f"
        return 0
    done < <(lease_holders)
    return 1
}

# Free MiB of the least-free GPU of the lease (sysfs only).
lease_min_free_mib() {
    local bus dev path used total free min=""
    for bus in ${EXPECT_PCI_BUSES//,/ }; do
        path=""
        for dev in "$DRM_SYSFS"/card*/device; do
            [[ -r $dev/mem_info_vram_used && -r $dev/mem_info_vram_total ]] || continue
            if [[ $(readlink -f "$dev") == *":$bus:"* ]]; then
                path=$dev
                break
            fi
        done
        if [[ -z $path ]]; then
            echo "lease: no mem_info_vram_used for PCI $bus under $DRM_SYSFS" >&2
            return 1
        fi
        used=$(cat "$path/mem_info_vram_used")
        total=$(cat "$path/mem_info_vram_total")
        free=$(((total - used) / 1024 / 1024))
        if [[ -z $min || $free -lt $min ]]; then
            min=$free
        fi
    done
    printf '%s\n' "$min"
}

# First field and the recorded pid of a holder/request line
# ("<owner> pid <pid> since <date>").
lease_line_owner() {
    awk 'NR==1{print $1}' "$1" 2>/dev/null
}

lease_line_pid() {
    awk 'NR==1{for (i = 1; i < NF; i++) if ($i == "pid") { print $(i + 1); exit }}' \
        "$1" 2>/dev/null
}

# True when the process recorded in a holder/request file is still alive.
lease_pid_alive() {
    local pid
    pid=$(lease_line_pid "$1")
    [[ -n $pid ]] && kill -0 "$pid" 2>/dev/null
}

# Remove our request (only while it is still ours) and our holder (only if this
# process wrote it).
lease_release() {
    if [[ $LEASE_INHERIT_ACTIVE == 1 ]]; then
        # Never delete the owner's holder: the outer driver releases it.
        LEASE_REQUEST=
        LEASE_HOLDER_CREATED=
        return 0
    fi
    if [[ -n $LEASE_REQUEST ]] &&
        [[ $(lease_line_owner "$LEASE_REQUEST") == "$OWNER" ]]; then
        rm -f "$LEASE_REQUEST"
    fi
    if [[ -n $LEASE_HOLDER_CREATED ]]; then
        rm -f "$LEASE_DIR/$OWNER.$SLOT" "$LEASE_DIR/$OWNER.$SLOT.gate"
    fi
    LEASE_REQUEST=
    LEASE_HOLDER_CREATED=
}

# True when the outer owner's live holder and acquisition-time free gate make
# this child boot safe to run without its own request/holder.
lease_inherit_check() {
    local holder="$LEASE_DIR/$OWNER.$SLOT" gate free
    [[ -e $holder ]] || return 1
    [[ $(lease_line_owner "$holder") == "$OWNER" ]] || return 1
    lease_pid_alive "$holder" || return 1
    gate=$(cat "$LEASE_DIR/$OWNER.$SLOT.gate" 2>/dev/null) || gate=
    [[ -n $gate ]] || return 1
    free=$gate
    ((free >= LEASE_FREE_GIB * 1024)) || return 1
    echo "lease: inheriting our $OWNER.$SLOT (gate ${free} MiB free per GPU at acquisition)"
    return 0
}

lease_refuse_foreign() {
    local who=${1:-lease} f
    if f=$(lease_foreign_holder); then
        echo "$who: $(basename "$f") holds the $SLOT lease; this kit is $OWNER.$SLOT" >&2
        echo "$who: wait for/release per $LEASE_DIR/README.md; do not delete another agent's file" >&2
        return 1
    fi
    return 0
}

# Refuse a foreign holder, then wait for a free lease and write our holder.
# The request file is one per slot (lane README): another owner's is never
# overwritten, we queue behind it.  An own holder is reused only while its pid is
# alive and the GPUs still show LEASE_FREE_GIB free; a stale one is dropped.
lease_acquire() {
    local f free want deadline req_owner request_blocked own_holder
    if [[ $LEASE_INHERIT == 1 ]] && lease_inherit_check; then
        LEASE_INHERIT_ACTIVE=1
        return 0
    fi
    lease_refuse_foreign "serve.sh" || return 1
    mkdir -p "$LEASE_DIR"
    # Armed before the request is written; serve.sh later replaces both traps
    # with its combined cleanup (server kill + lease_release).
    trap lease_release EXIT
    trap 'lease_release; exit 130' TERM INT
    want=$((LEASE_FREE_GIB * 1024))
    deadline=$((SECONDS + LEASE_WAIT_SECONDS))
    while :; do
        request_blocked=
        own_holder=
        if [[ -e $LEASE_DIR/$SLOT.request ]]; then
            req_owner=$(lease_line_owner "$LEASE_DIR/$SLOT.request")
            if [[ $req_owner != "$OWNER" ]]; then
                request_blocked=1
                echo "lease: $SLOT.request belongs to $req_owner; queueing (${LEASE_POLL_SECONDS} s)" >&2
            elif lease_pid_alive "$LEASE_DIR/$SLOT.request"; then
                if [[ $(lease_line_pid "$LEASE_DIR/$SLOT.request") == "$$" ]]; then
                    : # our own request from an earlier iteration: wait on the gate
                else
                    request_blocked=1
                    echo "lease: our own $SLOT.request is live (pid $(lease_line_pid "$LEASE_DIR/$SLOT.request")); queueing (${LEASE_POLL_SECONDS} s)" >&2
                fi
            else
                rm -f "$LEASE_DIR/$SLOT.request"
            fi
        fi
        if [[ -e $LEASE_DIR/$OWNER.$SLOT ]]; then
            if lease_pid_alive "$LEASE_DIR/$OWNER.$SLOT"; then
                own_holder=1
            else
                echo "lease: dropping stale holder $OWNER.$SLOT (pid $(lease_line_pid "$LEASE_DIR/$OWNER.$SLOT") is gone)" >&2
                rm -f "$LEASE_DIR/$OWNER.$SLOT"
            fi
        fi
        if [[ -z $request_blocked && -z $LEASE_REQUEST ]]; then
            LEASE_REQUEST="$LEASE_DIR/$SLOT.request"
            printf '%s pid %s since %s\n' "$OWNER" "$$" "$(date -Is)" >"$LEASE_REQUEST"
        fi
        if [[ -n $request_blocked ]]; then
            : # the queue message above is this iteration's wait reason
        elif f=$(lease_foreign_holder); then
            echo "lease: $(basename "$f") holds $SLOT; waiting (${LEASE_POLL_SECONDS} s)" >&2
        else
            free=$(lease_min_free_mib) || free=
            if [[ -n $free ]] && ((free >= want)); then
                if [[ -n $own_holder ]]; then
                    echo "lease: reusing our holder $OWNER.$SLOT (${free} MiB free per GPU)"
                else
                    printf '%s pid %s since %s\n' "$OWNER" "$$" "$(date -Is)" >"$LEASE_DIR/$OWNER.$SLOT"
                    printf '%s\n' "$free" >"$LEASE_DIR/$OWNER.$SLOT.gate"
                    LEASE_HOLDER_CREATED=1
                    echo "lease: took $OWNER.$SLOT (${free} MiB free per GPU)"
                fi
                if [[ -n $LEASE_REQUEST ]]; then
                    rm -f "$LEASE_REQUEST"
                fi
                LEASE_REQUEST=
                return 0
            fi
            if [[ -n $free ]]; then
                echo "lease: ${free} MiB free per GPU (< ${want} MiB); waiting (${LEASE_POLL_SECONDS} s)" >&2
            fi
        fi
        if ((SECONDS >= deadline)); then
            echo "lease: no free $SLOT window after ${LEASE_WAIT_SECONDS} s" >&2
            if [[ -n $LEASE_REQUEST ]] &&
                [[ $(lease_line_owner "$LEASE_REQUEST") == "$OWNER" ]]; then
                rm -f "$LEASE_REQUEST"
            fi
            LEASE_REQUEST=
            return 1
        fi
        sleep "$LEASE_POLL_SECONDS"
    done
}
