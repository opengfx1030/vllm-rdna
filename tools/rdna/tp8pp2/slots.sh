# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Harness settings + slot definitions for 4- and 8-GPU runs on par1-cs25. Source me.
#
#   H          directory of these scripts (not the run directory)
#   R          run directory: arms/, caches, abort markers   (RUN_DIR, default
#              ~/w4a8_runs/tp8pp2)
#   LANES      lease protocol root (default ~/w4a8_runs/lanes): lease/<owner>.slotA|B
#              holder files, quiet/<owner> measured-window files; BOOT_LOCK is the
#              flock file every boot takes
#   VENV       the serving venv (default ~/Apps/vllm/venv-7.14.0)
#
# Slot A = HIP 1-4 = V620 ce d3 d8 db (root c0); slot B = HIP 6-9 = V620 0a 0f 14 17
# (root 00); AB = both, in that order: ranks 0-3 on switch c0, ranks 4-7 on switch 00.
# HIP 0 and HIP 5 (the W7800s) are never part of a slot. Every GPU command sets
# HIP_VISIBLE_DEVICES on the same line. The HIP/bus maps are par1-cs25's; override
# SLOT_A_HIP / SLOT_A_BUSES / SLOT_B_HIP / SLOT_B_BUSES for another host.
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
slot_hip() {
    case $1 in
        A) echo "$SLOT_A_HIP" ;;
        B) echo "$SLOT_B_HIP" ;;
        AB) echo "$SLOT_A_HIP,$SLOT_B_HIP" ;;
        *) return 1 ;;
    esac
}
slot_buses() {
    case $1 in
        A) echo "$SLOT_A_BUSES" ;;
        B) echo "$SLOT_B_BUSES" ;;
        AB) echo "$SLOT_A_BUSES $SLOT_B_BUSES" ;;
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
verify_slot() {
    local s=$1 got i=0 hip bus nm n
    local -a want
    read -r -a want <<< "$(slot_buses "$s")"
    n=${#want[@]}
    got=$(resolve_slot "$s") || return 1
    while IFS=: read -r hip bus nm _; do
        if [[ $bus != "${want[$i]}" || $nm != *V620* ]]; then
            echo "REFUSE: HIP $hip is bus $bus ($nm), expected ${want[$i]} V620"
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
