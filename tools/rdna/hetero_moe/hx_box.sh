# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Box-rule helpers of the hx GPU steps on par1-cs25. Source me.
#
#   HX_OWNER   holder name in lease file names (default HxServer): lease/<owner>.a2
#              (HIP 0-1, buses d1 d4) and lease/<owner>.b4 (HIP 2-5, buses d9 dc df e2),
#              the interim six-V620 layout of the 2026-11-08 case move
#   hx_take_lease KIND [WHAT]   KIND a2 | b4. Refuses while another holder
#                               file of that kind exists or a GPU of it has more than
#                               HX_SLOT_BUSY_GIB of VRAM in use (an untracked job);
#                               writes the holder file.
#   hx_quiet_wait [MAX_SECONDS] waits until no quiet/<agent> file is newer than 6 min
#   hx_boot_run CMD...          quiet check, flock boot.lock, quiet re-check, run CMD
#   hx_quiet_begin / hx_quiet_end  a measured window: touch quiet/<owner>, wait until
#                               no boot holds boot.lock, keep the file fresh (re-touch
#                               every 2 min, for 15 min at most); end removes it
#   hx_evict_start PATTERN OUT  sample the KFD evicted_ms of the processes matching
#                               the pgrep pattern every 0.1 s into OUT
#   hx_evict_stop               stop; print evicted_ms gained per pid (a measured
#                               window with a jump is INVALID, box rules)
#   hx_verify_hips HIP...       every HIP index is the expected V620 (HIP 0-5 = bus
#                               d1 d4 d9 dc df e2), one torch process per index
#   hx_vram_used_gib BUS        VRAM in use (GiB) on the GPU at PCI bus BUS (two hex
#                               digits). rocm-smi indices are KFD order, not always
#                               HIP indices: always go through the bus.
#
# Rules (lease README, box rules): hold the lease of every leaf you use; every GPU
# command sets HIP_VISIBLE_DEVICES on its own line; a boot takes boot.lock and waits
# for quiet files older than 6 min; a measured window touches quiet/<owner> (re-touch
# every 5 min, at most 15 min).
#
# shellcheck shell=bash
HX_BOX_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=../tp8pp2/slots.sh
source "$HX_BOX_DIR/../tp8pp2/slots.sh"
HX_OWNER=${HX_OWNER:-HxServer}
HX_SLOT_BUSY_GIB=${HX_SLOT_BUSY_GIB:-2}
# Interim box (2026-11-08 case move): HIP 0-5 = V620 d1 d4 d9 dc df e2 (all gfx1030)
HX_HIP_BUSES=(d1 d4 d9 dc df e2)

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
        a2) buses="d1 d4" ;;
        b4) buses="d9 dc df e2" ;;
        *) hx_log "unknown lease kind $kind (a2 | b4)"; return 1 ;;
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
            [[ ${q##*/} == "$HX_OWNER" ]] && continue # our own measured window
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

hx_evict_start() {
    local pat=$1 out=$2
    (
        while :; do
            for p in $(pgrep -f "$pat"); do
                s=0
                for f in /sys/class/kfd/kfd/proc/"$p"/stats_*/evicted_ms; do
                    [[ -r $f ]] && s=$((s + $(cat "$f" 2>/dev/null || echo 0)))
                done
                echo "$(date +%s.%N) $p $s"
            done
            sleep 0.1
        done
    ) >"$out" 2>/dev/null &
    HX_EVICT_PID=$!
    HX_EVICT_OUT=$out
}

hx_evict_stop() {
    [[ -n ${HX_EVICT_PID:-} ]] || return 0
    kill "$HX_EVICT_PID" 2>/dev/null
    wait "$HX_EVICT_PID" 2>/dev/null
    HX_EVICT_PID=""
    awk '{ if (!($2 in f)) f[$2] = $3; l[$2] = $3 }
         END { for (p in f) printf "evicted_ms pid %s: %d\n", p, l[p] - f[p] }' "$HX_EVICT_OUT"
}

hx_verify_hips() {
    local hip out want
    for hip in "$@"; do
        [[ $hip =~ ^[0-5]$ ]] || {
            echo "REFUSE: HIP '$hip' is not one of the six V620s (0-5)"
            return 1
        }
        want=${HX_HIP_BUSES[hip]}
        out=$(HIP_VISIBLE_DEVICES=$hip "$VENV/bin/python" -c "
import torch
p = torch.cuda.get_device_properties(0)
print('%02x %s' % (p.pci_bus_id, p.name))
") || return 1
        if [[ $out != "$want "*V620* ]]; then
            echo "REFUSE: HIP $hip is '$out', expected bus $want V620"
            return 1
        fi
    done
    echo "hips verified: $*"
}
