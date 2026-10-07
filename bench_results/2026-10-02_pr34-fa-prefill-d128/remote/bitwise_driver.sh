#!/usr/bin/env bash
set -uo pipefail
V=/home/chenco_adm/Apps/vllm/venv-7.14.0_0.28.0
T=/home/chenco_adm/vllm-rdna-0.28.0
D=/home/chenco_adm/w4a8_runs/pr34_validate
SL=$V/lib/python3.12/site-packages/_rocm_sdk_libraries/lib
SC=$V/lib/python3.12/site-packages/_rocm_sdk_core/lib
export LD_LIBRARY_PATH="$SL:$SC/host-math/lib:$SC/rocm_sysdeps/lib:$SC/core/lib:$V/lib/python3.12/site-packages/torch/lib"
export HIP_VISIBLE_DEVICES=4
for arm in baseline patched; do
  cp "$D/_rocm_C.$arm.so" "$T/vllm/_rocm_C.abi3.so"
  "$V/bin/python" "$D/probe_bitwise_gqa256.py" "$D/bw_$arm.pt"
done
"$V/bin/python" - <<'PY'
import torch
d="/home/chenco_adm/w4a8_runs/pr34_validate"
a=torch.load(f"{d}/bw_baseline.pt"); b=torch.load(f"{d}/bw_patched.pt")
print("D256_EVEN_BITWISE_IDENTICAL:", torch.equal(a,b))
print("max_abs_diff:", (a.float()-b.float()).abs().max().item())
PY
