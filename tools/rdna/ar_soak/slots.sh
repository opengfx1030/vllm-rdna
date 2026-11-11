# shellcheck shell=bash
# Slot definitions + HIP -> PCI bus resolution (lease README 2026-11-05 15:35).
# Source me. Every GPU command sets HIP_VISIBLE_DEVICES on the same line.
VENV=${VENV:-$HOME/Apps/vllm/venv-7.14.0}
SITE=$(echo "$VENV"/lib/python3*/site-packages)
export LD_LIBRARY_PATH=$SITE/_rocm_sdk_libraries/lib:$SITE/_rocm_sdk_core/lib/host-math/lib:$SITE/_rocm_sdk_core/lib/rocm_sysdeps/lib:$SITE/_rocm_sdk_core/lib:$SITE/torch/lib
slot_hip() { case $1 in A) echo 1,2,3,4 ;; B) echo 6,7,8,9 ;; *) return 1 ;; esac; }
slot_buses() { case $1 in A) echo "ce d3 d8 db" ;; B) echo "0a 0f 14 17" ;; *) return 1 ;; esac; }
# resolve_slot SLOT -> "hip:bus:name:GiB" per visible index
resolve_slot() {
    local s=$1 hips
    hips=$(slot_hip "$s") || return 1
    HIP_VISIBLE_DEVICES=$hips "$VENV/bin/python" -c "
import torch
hips = \"$hips\".split(\",\")
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print(f\"{hips[i]}:{p.pci_bus_id:02x}:{p.name.replace(\" \", \"_\")}:{p.total_memory/2**30:.1f}\")
"
}
# verify_slot SLOT: exit 0 only if every leased index is the expected V620, in order
verify_slot() {
    local s=$1 want got i=0 hip bus nm
    read -r -a want <<< "$(slot_buses "$s")"
    got=$(resolve_slot "$s") || return 1
    while IFS=: read -r hip bus nm _; do
        if [[ $bus != "${want[$i]}" || $nm != *V620* ]]; then
            echo "REFUSE: HIP $hip is bus $bus ($nm), expected ${want[$i]} V620"
            return 1
        fi
        i=$((i + 1))
    done <<< "$got"
    ((i == 4)) || { echo "REFUSE: $i devices visible, expected 4"; return 1; }
    echo "slot $s verified: $(echo "$got" | tr '\n' ' ')"
}
