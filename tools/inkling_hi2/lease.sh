#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Shared GPU-lease carrier for the Inkling-2 HI2 kit (AGENTS.md, "Two-slot box
# operation"; ~/w4a8_runs/lanes/lease/README.md):
#
#   holder files <owner>.<slot>; the next job writes <slot>.request (trap armed
#   before the request), waits until no other holder exists and every GPU of the
#   lease has >= 28 GiB free, then writes its holder; the holder is removed by
#   the release trap.
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

# Remove our request (always) and our holder (only if this process wrote it).
lease_release() {
    [[ -n $LEASE_REQUEST ]] && rm -f "$LEASE_REQUEST"
    if [[ -n $LEASE_HOLDER_CREATED ]]; then
        rm -f "$LEASE_DIR/$OWNER.$SLOT"
    fi
    LEASE_REQUEST=
    LEASE_HOLDER_CREATED=
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
lease_acquire() {
    local f free want deadline
    lease_refuse_foreign "serve.sh" || return 1
    if [[ -e $LEASE_DIR/$OWNER.$SLOT ]]; then
        echo "lease: reusing our holder $OWNER.$SLOT"
        return 0
    fi
    mkdir -p "$LEASE_DIR"
    # Armed before the request is written; serve.sh later replaces both traps
    # with its combined cleanup (server kill + lease_release).
    trap lease_release EXIT
    trap 'lease_release; exit 130' TERM INT
    LEASE_REQUEST="$LEASE_DIR/$SLOT.request"
    printf '%s pid %s since %s\n' "$OWNER" "$$" "$(date -Is)" >"$LEASE_REQUEST"
    want=$((LEASE_FREE_GIB * 1024))
    deadline=$((SECONDS + LEASE_WAIT_SECONDS))
    while :; do
        if f=$(lease_foreign_holder); then
            echo "lease: $(basename "$f") holds $SLOT; waiting (${LEASE_POLL_SECONDS} s)" >&2
        elif free=$(lease_min_free_mib); then
            if ((free >= want)); then
                break
            fi
            echo "lease: ${free} MiB free per GPU (< ${want} MiB); waiting (${LEASE_POLL_SECONDS} s)" >&2
        fi
        if ((SECONDS >= deadline)); then
            echo "lease: no free $SLOT window after ${LEASE_WAIT_SECONDS} s" >&2
            rm -f "$LEASE_REQUEST"
            LEASE_REQUEST=
            return 1
        fi
        sleep "$LEASE_POLL_SECONDS"
    done
    printf '%s pid %s since %s\n' "$OWNER" "$$" "$(date -Is)" >"$LEASE_DIR/$OWNER.$SLOT"
    LEASE_HOLDER_CREATED=1
    rm -f "$LEASE_REQUEST"
    LEASE_REQUEST=
    echo "lease: took $OWNER.$SLOT (${free} MiB free per GPU)"
}
