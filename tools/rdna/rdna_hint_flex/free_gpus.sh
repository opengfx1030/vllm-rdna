#!/usr/bin/env bash
# Print the HIP indices among 2-5 that are free: no lock file naming them
# (w4a8_runs/hadamard-int2/*.gpuN.lock), and 0% use and 0% VRAM in three
# rocm-smi samples 5 s apart. rocm-smi GPU[i+2] is HIP i on par1-cs25.
LOCKS=${LOCKS:-$HOME/w4a8_runs/hadamard-int2}
free=""
for hip in ${HIPS:-2 3 4 5}; do
  ls "$LOCKS"/*.gpu"$hip".lock > /dev/null 2>&1 && continue
  ok=1
  for _ in 1 2 3; do
    use=$(rocm-smi --showuse | grep -E "GPU\[$((hip + 2))\]" | grep -oE "[0-9]+$")
    mem=$(rocm-smi --showmemuse | grep -E "GPU\[$((hip + 2))\].*VRAM" | grep -oE "[0-9]+$")
    if [[ ${use:-1} != 0 || ${mem:-1} != 0 ]]; then ok=0; break; fi
    sleep 5
  done
  (( ok )) && free="$free $hip"
done
echo "${free# }"
