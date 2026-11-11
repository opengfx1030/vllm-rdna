# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Harness settings + slot definitions for 4-, 5- and 8-GPU runs on par1-cs25. Source me.
#
#   H          directory of these scripts (not the run directory)
#   R          run directory: arms/, caches, abort markers   (RUN_DIR, default
#              ~/w4a8_runs/tp8pp2)
#   LANES      lease protocol root (default ~/w4a8_runs/lanes): lease/<owner>.slotA|B (10-GPU
#              layout) and <owner>.a2|b4 (interim layout) holder files, quiet/<owner>
#              measured-window files; BOOT_LOCK is the flock file every boot takes
#   VENV       the serving venv (default ~/Apps/vllm/venv-7.14.0)
#
# Slot A = HIP 1-4 = V620 ce d3 d8 db (root c0); slot B = HIP 6-9 = V620 0a 0f 14 17
# (root 00); AB = both, in that order: ranks 0-3 on switch c0, ranks 4-7 on switch 00.
# HIP 0 and HIP 5 (the W7800s) are never part of slots A, B or AB; slot HA (CFG=h1) adds
# HIP 0 (bus cb) as the hub in front of slot A. These describe the 10-GPU layout BEFORE the
# 2026-11-08 case move; verify_slot refuses them on today's box (box rules, INTERIM).
# Slot HV (CFG=h1v) is the interim layout's 5-GPU job: six V620s, lease a2 = HIP 0,1 = buses
# d1 d4 (leaf A), lease b4 = HIP 2-5 = buses d9 dc df e2 (leaf B); the hub on HIP 1 (bus d4,
# lease a2) in front of the four servers on HIP 2-5 (lease b4); HIP 0 stays out of it.
# Every GPU command sets HIP_VISIBLE_DEVICES on the same line. The HIP/bus maps are
# par1-cs25's; override SLOT_A_HIP / SLOT_A_BUSES / SLOT_B_HIP / SLOT_B_BUSES / HUB_* /
# STANDIN_* for another host.
# shellcheck disable=SC2034  # H, R, LEASE, QUIET ... are read by the scripts that source this
H=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
R=${RUN_DIR:-$HOME/w4a8_runs/tp8pp2}
LANES=${LANES:-$HOME/w4a8_runs/lanes}
LEASE=$LANES/lease
QUIET=$LANES/quiet
BOOT_LOCK=${BOOT_LOCK:-$HOME/w4a8_runs/boot.lock}
export VENV=${VENV:-$HOME/Apps/vllm/venv-7.14.0}
SITE=$(echo "$VENV"/lib/python3*/site-packages)
export LD_LIBRARY_PATH=$SITE/_rocm_sdk_libraries/lib:$SITE/_rocm_sdk_core/lib:$SITE/_rocm_sdk_core/lib/host-math/lib:$SITE/_rocm_sdk_core/lib/rocm_sysdeps/lib:$SITE/torch/lib
SLOT_A_HIP=${SLOT_A_HIP:-1,2,3,4}
SLOT_A_BUSES=${SLOT_A_BUSES:-"ce d3 d8 db"}
SLOT_B_HIP=${SLOT_B_HIP:-6,7,8,9}
SLOT_B_BUSES=${SLOT_B_BUSES:-"0a 0f 14 17"}
# HA = hub + slot A (hetero-MoE H1): the W7800 of island 0 first, then slot A. HIP 5 /
# bus 07 (the other W7800) is never touched.
HUB_HIP=${HUB_HIP:-0}
HUB_BUSES=${HUB_BUSES:-cb}
# HV = the V620 stand-in of the hetero MoE (CFG=h1v): hub, then servers; STANDIN_LEASES are
# the lease names it takes (lease/<owner>.<name>).
STANDIN_HUB_HIP=${STANDIN_HUB_HIP:-1}
STANDIN_HUB_BUSES=${STANDIN_HUB_BUSES:-d4}
STANDIN_SERVER_HIP=${STANDIN_SERVER_HIP:-2,3,4,5}
STANDIN_SERVER_BUSES=${STANDIN_SERVER_BUSES:-"d9 dc df e2"}
STANDIN_LEASES=${STANDIN_LEASES:-"a2 b4"}
slot_hip() {
    case $1 in
        A) echo "$SLOT_A_HIP" ;;
        B) echo "$SLOT_B_HIP" ;;
        AB) echo "$SLOT_A_HIP,$SLOT_B_HIP" ;;
        HA) echo "$HUB_HIP,$SLOT_A_HIP" ;;
        HV) echo "$STANDIN_HUB_HIP,$STANDIN_SERVER_HIP" ;;
        *) return 1 ;;
    esac
}
slot_buses() {
    case $1 in
        A) echo "$SLOT_A_BUSES" ;;
        B) echo "$SLOT_B_BUSES" ;;
        AB) echo "$SLOT_A_BUSES $SLOT_B_BUSES" ;;
        HA) echo "$HUB_BUSES $SLOT_A_BUSES" ;;
        HV) echo "$STANDIN_HUB_BUSES $STANDIN_SERVER_BUSES" ;;
        *) return 1 ;;
    esac
}
# resolve_slot SLOT -> prints "hip:bus:name:GiB" lines for the slot list
# (visible index i -> torch pci_bus_id)
resolve_slot() {
    local s=$1 hips
    hips=$(slot_hip "$s") || return 1
    HIP_VISIBLE_DEVICES=$hips "$VENV/bin/python" -c "
import torch
hips = \"$hips\".split(\",\")
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print(f\"{hips[i]}:{p.pci_bus_id:02x}:{p.name.replace(' ', '_')}:{p.total_memory/2**30:.1f}\")
"
}
# verify_slot SLOT: exit 0 only if every leased index is the expected V620, in order
# (slot HA: the first device must instead be the W7800 on HUB_BUSES)
verify_slot() {
    local s=$1 got i=0 hip bus nm n need
    local -a want
    read -r -a want <<< "$(slot_buses "$s")"
    n=${#want[@]}
    got=$(resolve_slot "$s") || return 1
    while IFS=: read -r hip bus nm _; do
        need=V620
        [[ $s == HA && $i == 0 ]] && need=W7800
        if [[ $bus != "${want[$i]}" || $nm != *"$need"* ]]; then
            echo "REFUSE: HIP $hip is bus $bus ($nm), expected ${want[$i]} $need"
            return 1
        fi
        i=$((i + 1))
    done <<< "$got"
    ((i == n)) || {
        echo "REFUSE: $i devices visible, expected $n"
        return 1
    }
    echo "slot $s verified: $(echo "$got" | tr '\n' ' ')"
}
