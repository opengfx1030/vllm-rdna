# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Box-rule helpers of the hx GPU steps on par1-cs25. Source me.
#
#   HX_OWNER   holder name in lease file names (default HxServer):
#              lease/<owner>.slotA, lease/<owner>.w7800cb
#   hx_take_lease KIND [WHAT]   KIND slotA | w7800cb. Refuses while another holder
#                               file of that kind exists or a GPU of it has more than
#                               HX_SLOT_BUSY_GIB of VRAM in use (an untracked job);
#                               writes the holder file.
#   hx_release_lease KIND
#   hx_quiet_wait [MAX_SECONDS] waits until no quiet/<agent> file is newer than 6 min
#   hx_boot_run CMD...          quiet check, flock boot.lock, quiet re-check, run CMD
#   hx_quiet_begin / hx_quiet_end  a measured window: touch quiet/<owner>, wait until
#                               no boot holds boot.lock, keep the file fresh (re-touch
#                               every 2 min, for 15 min at most); end removes it
#   hx_verify_hub               HIP 0 is bus cb and a W7800
#   hx_verify_slot_a            HIP 1-4 are the V620 ce d3 d8 db (slots.sh)
#   hx_vram_used_gib BUS        VRAM in use (GiB) on the GPU at PCI bus BUS (two hex
#                               digits). rocm-smi indices are KFD order (07 0a 0f 14
#                               17 cb ce d3 d8 db), NOT HIP indices: always go through
#                               the bus.
#
# Rules (lease README, box rules): HIP 5 never; the W7800 only with the w7800cb
# lease; every GPU command sets HIP_VISIBLE_DEVICES on its own line; a boot takes
# boot.lock and waits for quiet files older than 6 min; a measured window touches
# quiet/<owner> (re-touch every 5 min, at most 15 min).
#
# shellcheck shell=bash
HX_BOX_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=../tp8pp2/slots.sh
source "$HX_BOX_DIR/../tp8pp2/slots.sh"
HX_OWNER=${HX_OWNER:-HxServer}
HX_SLOT_BUSY_GIB=${HX_SLOT_BUSY_GIB:-2}

hx_log() { echo "[hx $(date +%T)] $*"; }

hx_vram_used_gib() {
    local bus=${1,,} idx
    idx=$(rocm-smi --showbus 2>/dev/null |
        awk -v b="0000:${bus^^}:" 'index($0, b) {match($0, /GPU\[[0-9]+\]/); print substr($0, RSTART + 4, RLENGTH - 5); exit}')
    [[ -n $idx ]] || return 1
    rocm-smi --showmeminfo vram 2>/dev/null |
        awk -v g="GPU[$idx]" 'index($0, g) && /Used Memory/ {printf "%.2f\n", $NF / 2^30; exit}'
}

hx_take_lease() {
    local kind=$1 what=${2:-hx} f other bus used buses
    mkdir -p "$LEASE"
    for f in "$LEASE"/*."$kind"; do
        [[ -e $f ]] || continue
        [[ $f == "$LEASE/$HX_OWNER.$kind" ]] && continue
        other=$(cat "$f" 2>/dev/null)
        hx_log "lease $kind is held: ${f##*/}: $other"
        return 1
    done
    case $kind in
        slotA) buses=$SLOT_A_BUSES ;;
        w7800cb) buses=cb ;;
        *) hx_log "unknown lease kind $kind"; return 1 ;;
    esac
    for bus in $buses; do
        used=$(hx_vram_used_gib "$bus") || {
            hx_log "$kind: bus $bus not found in rocm-smi"
            return 1
        }
        if awk -v u="$used" -v m="$HX_SLOT_BUSY_GIB" 'BEGIN {exit !(u > m)}'; then
            hx_log "$kind: bus $bus has ${used} GiB in use by a job without a lease file"
            return 1
        fi
    done
    (
        set -o noclobber
        echo "$HX_OWNER $(date -Iseconds) $what pid $$" >"$LEASE/$HX_OWNER.$kind"
    ) 2>/dev/null || [[ -f $LEASE/$HX_OWNER.$kind ]] || return 1
    rm -f "$LEASE/${kind}.request"
    hx_log "lease $kind taken ($what)"
}

hx_release_lease() {
    rm -f "$LEASE/$HX_OWNER.$1"
    hx_log "lease $1 released"
}

hx_quiet_wait() {
    local max=${1:-1800} t0 q newest
    t0=$(date +%s)
    while :; do
        newest=""
        for q in "$QUIET"/*; do
            [[ -e $q ]] || continue
            if [[ -n $(find "$q" -mmin -6 2>/dev/null) ]]; then newest=$q; fi
        done
        [[ -z $newest ]] && return 0
        (($(date +%s) - t0 > max)) && {
            hx_log "quiet window of ${newest##*/} still open after ${max}s"
            return 1
        }
        sleep 20
    done
}

hx_boot_run() {
    hx_quiet_wait || return 1
    (
        flock -x 9
        hx_quiet_wait 60 || exit 1
        "$@"
    ) 9>"$BOOT_LOCK"
}

hx_quiet_begin() {
    local t0 now
    mkdir -p "$QUIET"
    touch "$QUIET/$HX_OWNER"
    t0=$(date +%s)
    until flock -n "$BOOT_LOCK" true; do
        now=$(date +%s)
        if ((now - t0 > 600)); then
            rm -f "$QUIET/$HX_OWNER"
            hx_log "boot.lock still held after 600 s"
            return 1
        fi
        sleep 2
    done
    (
        for _ in 1 2 3 4 5 6 7; do
            sleep 120
            touch "$QUIET/$HX_OWNER"
        done
    ) &
    HX_QUIET_PID=$!
    hx_log "quiet window open (15 min at most)"
}

hx_quiet_end() {
    if [[ -n ${HX_QUIET_PID:-} ]]; then
        kill "$HX_QUIET_PID" 2>/dev/null
        wait "$HX_QUIET_PID" 2>/dev/null
        HX_QUIET_PID=""
    fi
    rm -f "$QUIET/$HX_OWNER"
    hx_log "quiet window closed"
}

hx_verify_hub() {
    local out
    out=$(HIP_VISIBLE_DEVICES=0 "$VENV/bin/python" -c "
import torch
p = torch.cuda.get_device_properties(0)
print('%02x %s' % (p.pci_bus_id, p.name))
") || return 1
    if [[ $out != cb\ *W7800* ]]; then
        echo "REFUSE: HIP 0 is '$out', expected bus cb W7800"
        return 1
    fi
    echo "hub verified: HIP 0 = $out"
}

hx_verify_slot_a() { verify_slot A; }
