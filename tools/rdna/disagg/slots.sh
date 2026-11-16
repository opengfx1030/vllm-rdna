# Lease definitions + HIP->PCI-bus resolution (lease README INTERIM layout, 2026-11-10:
# six V620s behind one root port; b4 = TP4 group HIP 2-5, a2 = HIP 0,1).
# Source me. Everything runs with HIP_VISIBLE_DEVICES set on the same line.
VENV=${VENV:-$HOME/Apps/vllm/venv-7.14.0}
SITE=$(echo "$VENV"/lib/python3*/site-packages)
export LD_LIBRARY_PATH=$SITE/_rocm_sdk_libraries/lib:$SITE/_rocm_sdk_core/lib/host-math/lib:$SITE/_rocm_sdk_core/lib/rocm_sysdeps/lib:$SITE/_rocm_sdk_core/lib/core/lib:$SITE/torch/lib
slot_hip() { case $1 in b4) echo 2,3,4,5 ;; a2) echo 0,1 ;; *) return 1 ;; esac; }
slot_buses() { case $1 in b4) echo "d9 dc df e2" ;; a2) echo "d1 d4" ;; *) return 1 ;; esac; }
# resolve_slot SLOT -> prints "hip:bus:name:GiB" per slot GPU (visible idx i -> torch pci_bus_id)
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
    local s=$1 got i=0 hip bus nm _mem n
    local -a want
    read -r -a want <<< "$(slot_buses "$s")"
    n=${#want[@]}
    got=$(resolve_slot "$s") || return 1
    while IFS=: read -r hip bus nm _mem; do
        if [[ $bus != "${want[$i]}" || $nm != *V620* ]]; then
            echo "REFUSE: HIP $hip is bus $bus ($nm), expected ${want[$i]} V620"
            return 1
        fi
        i=$((i + 1))
    done <<< "$got"
    ((i == n)) || { echo "REFUSE: $i devices visible, expected $n"; return 1; }
    echo "slot $s verified: $(echo "$got" | tr "\n" " ")"
}
